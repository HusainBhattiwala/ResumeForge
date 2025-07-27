import gc
import json
import asyncio
import logging
import markdown
import numpy as np

from sqlalchemy.future import select
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Optional, Tuple, AsyncGenerator

from app.prompt import prompt_factory
from app.schemas.json import json_schema_factory
from app.schemas.pydantic import ResumePreviewerModel
from app.agent import EmbeddingManager, AgentManager
from app.models import Resume, Job, ProcessedResume, ProcessedJob
from .exceptions import (
    ResumeNotFoundError,
    JobNotFoundError,
    ResumeParsingError,
    JobParsingError,
    ResumeKeywordExtractionError,
    JobKeywordExtractionError,
)

logger = logging.getLogger(__name__)


class ScoreImprovementService:
    """
    Service to handle scoring of resumes and jobs using embeddings.
    Fetches Resume and Job data from the database, computes embeddings,
    and calculates cosine similarity scores. Uses LLM for iteratively improving
    the scoring process.
    """

    def __init__(self, db: AsyncSession, max_retries: int = 5):
        self.db = db
        self.max_retries = max_retries
        self.md_agent_manager = AgentManager(strategy="md")
        self.json_agent_manager = AgentManager()
        self.embedding_manager = EmbeddingManager()

    def _validate_resume_keywords(
        self, processed_resume: ProcessedResume, resume_id: str
    ) -> None:
        """
        Validates that keyword extraction was successful for a resume.
        Raises ResumeKeywordExtractionError if keywords are missing or empty.
        """
        if not processed_resume.extracted_keywords:
            raise ResumeKeywordExtractionError(resume_id=resume_id)

        try:
            keywords_data = json.loads(processed_resume.extracted_keywords)
            if keywords_data is None:
                raise ResumeKeywordExtractionError(resume_id=resume_id)
            keywords = keywords_data.get("extracted_keywords", [])
            if not keywords or len(keywords) == 0:
                raise ResumeKeywordExtractionError(resume_id=resume_id)
        except (json.JSONDecodeError, AttributeError):
            raise ResumeKeywordExtractionError(resume_id=resume_id)

    def _validate_job_keywords(self, processed_job: ProcessedJob, job_id: str) -> None:
        """
        Validates that keyword extraction was successful for a job.
        Raises JobKeywordExtractionError if keywords are missing or empty.
        """
        if not processed_job.extracted_keywords:
            raise JobKeywordExtractionError(job_id=job_id)

        try:
            keywords_data = json.loads(processed_job.extracted_keywords)
            keywords = keywords_data.get("extracted_keywords", [])
            if not keywords or len(keywords) == 0:
                raise JobKeywordExtractionError(job_id=job_id)
        except json.JSONDecodeError:
            raise JobKeywordExtractionError(job_id=job_id)

    async def _get_resume(
        self, resume_id: str
    ) -> Tuple[Resume | None, ProcessedResume | None]:
        """
        Fetches the resume from the database.
        """
        query = select(Resume).where(Resume.resume_id == resume_id)
        result = await self.db.execute(query)
        resume = result.scalars().first()

        if not resume:
            raise ResumeNotFoundError(resume_id=resume_id)

        query = select(ProcessedResume).where(ProcessedResume.resume_id == resume_id)
        result = await self.db.execute(query)
        processed_resume = result.scalars().first()

        if not processed_resume:
            raise ResumeParsingError(resume_id=resume_id)

        self._validate_resume_keywords(processed_resume, resume_id)

        return resume, processed_resume

    async def _get_job(self, job_id: str) -> Tuple[Job | None, ProcessedJob | None]:
        """
        Fetches the job from the database.
        """
        query = select(Job).where(Job.job_id == job_id)
        result = await self.db.execute(query)
        job = result.scalars().first()

        if not job:
            raise JobNotFoundError(job_id=job_id)

        query = select(ProcessedJob).where(ProcessedJob.job_id == job_id)
        result = await self.db.execute(query)
        processed_job = result.scalars().first()

        if not processed_job:
            raise JobParsingError(job_id=job_id)

        self._validate_job_keywords(processed_job, job_id)

        return job, processed_job

    def calculate_cosine_similarity(
        self,
        extracted_job_keywords_embedding: np.ndarray,
        resume_embedding: np.ndarray,
    ) -> float:
        """
        Calculates the cosine similarity between two embeddings.
        """
        if resume_embedding is None or extracted_job_keywords_embedding is None:
            return 0.0

        ejk = np.asarray(extracted_job_keywords_embedding).squeeze()
        re = np.asarray(resume_embedding).squeeze()

        return float(np.dot(ejk, re) / (np.linalg.norm(ejk) * np.linalg.norm(re)))

    async def improve_score_with_llm(
        self,
        resume: str,
        extracted_resume_keywords: str,
        job: str,
        extracted_job_keywords: str,
        previous_cosine_similarity_score: float,
        extracted_job_keywords_embedding: np.ndarray,
    ) -> Tuple[str, float]:
        prompt_template = prompt_factory.get("resume_improvement")
        best_resume, best_score = resume, previous_cosine_similarity_score

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                f"Attempt {attempt}/{self.max_retries} to improve resume score."
            )
            prompt = prompt_template.format(
                raw_job_description=job,
                extracted_job_keywords=extracted_job_keywords,
                raw_resume=best_resume,
                extracted_resume_keywords=extracted_resume_keywords,
                current_cosine_similarity=best_score,
            )
            improved = await self.md_agent_manager.run(prompt)
            emb = await self.embedding_manager.embed(text=improved)
            score = self.calculate_cosine_similarity(
                emb, extracted_job_keywords_embedding
            )

            if score > best_score:
                return improved, score

            logger.info(
                f"Attempt {attempt} resulted in score: {score}, best score so far: {best_score}"
            )

        return best_resume, best_score

    async def generate_detailed_analysis(
        self,
        resume: str,
        extracted_resume_keywords: str,
        job: str,
        extracted_job_keywords: str,
        cosine_similarity_score: float,
    ) -> Dict:
        """
        Generate detailed resume analysis including gaps, strengths, and recommendations.
        """
        try:
            prompt_template = prompt_factory.get("resume_analysis")
            prompt = prompt_template.format(
                raw_job_description=job,
                extracted_job_keywords=extracted_job_keywords,
                raw_resume=resume,
                extracted_resume_keywords=extracted_resume_keywords,
                current_cosine_similarity=cosine_similarity_score,
            )
            
            analysis_response = await self.json_agent_manager.run(prompt=prompt)
            
            # Parse JSON response with cleaning
            if isinstance(analysis_response, str):
                import re
                
                # Clean the JSON string
                cleaned_response = analysis_response.strip()
                
                # Remove markdown code block markers if present
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                elif cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                
                # Handle incomplete JSON that starts with a field name
                if cleaned_response.startswith('"') and not cleaned_response.startswith('{'):
                    # Try to wrap in braces if it looks like it's missing the opening brace
                    cleaned_response = '{' + cleaned_response
                
                # Handle incomplete JSON that's missing closing brace
                if cleaned_response.startswith('{') and not cleaned_response.endswith('}'):
                    # Count braces to see if we need to add closing braces
                    open_braces = cleaned_response.count('{')
                    close_braces = cleaned_response.count('}')
                    missing_braces = open_braces - close_braces
                    if missing_braces > 0:
                        cleaned_response += '}' * missing_braces
                
                # Remove any trailing commas before closing braces/brackets
                cleaned_response = re.sub(r',(\s*[}\]])', r'\1', cleaned_response)
                
                # Strip again after cleaning
                cleaned_response = cleaned_response.strip()
                
                # Log the response for debugging
                logger.info(f"Attempting to parse analysis JSON: {cleaned_response[:200]}...")
                
                analysis = json.loads(cleaned_response)
            else:
                analysis = analysis_response
                
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in detailed analysis: {str(e)}")
            logger.error(f"Raw analysis response: {repr(analysis_response)}")
            if 'cleaned_response' in locals():
                logger.error(f"Cleaned response: {repr(cleaned_response)}")
            # Return fallback analysis instead of re-raising
            return self._get_fallback_analysis(cosine_similarity_score, str(e))
        except Exception as e:
            logger.error(f"Error generating detailed analysis: {str(e)}")
            # Return fallback analysis instead of re-raising
            return self._get_fallback_analysis(cosine_similarity_score, str(e))

    def _get_fallback_analysis(self, cosine_similarity_score: float, error_msg: str) -> Dict:
        """
        Returns a fallback analysis when JSON parsing fails.
        """
        return {
            "overall_match_score": f"{cosine_similarity_score * 100:.1f}%",
            "strengths": ["Analysis unavailable due to processing error"],
            "gaps": ["Unable to analyze gaps - please review manually"],
            "keyword_analysis": {
                "matched_keywords": [],
                "missing_keywords": [],
                "keyword_frequency": "Analysis unavailable"
            },
            "section_analysis": {
                "experience": {
                    "score": f"{cosine_similarity_score * 100:.1f}%",
                    "feedback": "Analysis unavailable"
                },
                "skills": {
                    "score": f"{cosine_similarity_score * 100:.1f}%",
                    "feedback": "Analysis unavailable"
                },
                "education": {
                    "score": f"{cosine_similarity_score * 100:.1f}%",
                    "feedback": "Analysis unavailable"
                }
            },
            "ats_compatibility": {
                "score": "Unknown",
                "issues": [],
                "recommendations": []
            },
            "improvement_recommendations": [
                {
                    "priority": "high",
                    "recommendation": "Review and manually compare resume with job requirements",
                    "impact": "Manual analysis recommended due to processing error"
                }
            ],
            "missing_skills": [],
            "buzzwords_analysis": {
                "good_buzzwords": [],
                "missing_buzzwords": [],
                "overused_terms": []
            },
            "error": error_msg
        }

    async def get_resume_for_previewer(self, updated_resume: str) -> Dict:
        """
        Returns the updated resume in a format suitable for the dashboard.
        """
        prompt_template = prompt_factory.get("structured_resume")
        prompt = prompt_template.format(
            json.dumps(json_schema_factory.get("resume_preview"), indent=2),
            updated_resume,
        )
        logger.info(f"Structured Resume Prompt: {prompt}")
        raw_output = await self.json_agent_manager.run(prompt=prompt)

        try:
            resume_preview: ResumePreviewerModel = ResumePreviewerModel.model_validate(
                raw_output
            )
        except ValidationError as e:
            logger.info(f"Validation error: {e}")
            return None
        return resume_preview.model_dump()

    async def run(self, resume_id: str, job_id: str) -> Dict:
        """
        Main method to run the scoring and improving process and return dict.
        """

        resume, processed_resume = await self._get_resume(resume_id)
        job, processed_job = await self._get_job(job_id)

        extracted_job_keywords = ", ".join(
            json.loads(processed_job.extracted_keywords).get("extracted_keywords", [])
        )

        extracted_resume_keywords = ", ".join(
            json.loads(processed_resume.extracted_keywords).get(
                "extracted_keywords", []
            )
        )

        resume_embedding_task = asyncio.create_task(
            self.embedding_manager.embed(resume.content)
        )
        job_kw_embedding_task = asyncio.create_task(
            self.embedding_manager.embed(extracted_job_keywords)
        )
        resume_embedding, extracted_job_keywords_embedding = await asyncio.gather(
            resume_embedding_task, job_kw_embedding_task
        )

        cosine_similarity_score = self.calculate_cosine_similarity(
            extracted_job_keywords_embedding, resume_embedding
        )
        updated_resume, updated_score = await self.improve_score_with_llm(
            resume=resume.content,
            extracted_resume_keywords=extracted_resume_keywords,
            job=job.content,
            extracted_job_keywords=extracted_job_keywords,
            previous_cosine_similarity_score=cosine_similarity_score,
            extracted_job_keywords_embedding=extracted_job_keywords_embedding,
        )

        resume_preview = await self.get_resume_for_previewer(
            updated_resume=updated_resume
        )

        logger.info(f"Resume Preview: {resume_preview}")

        # Generate detailed analysis
        analysis = await self.generate_detailed_analysis(
            resume=resume.content,
            extracted_resume_keywords=extracted_resume_keywords,
            job=job.content,
            extracted_job_keywords=extracted_job_keywords,
            cosine_similarity_score=cosine_similarity_score,
        )

        execution = {
            "resume_id": resume_id,
            "job_id": job_id,
            "original_score": cosine_similarity_score,
            "new_score": updated_score,
            "updated_resume": markdown.markdown(text=updated_resume),
            "resume_preview": resume_preview,
            "detailed_analysis": analysis,
        }

        gc.collect()

        return execution

    async def run_and_stream(self, resume_id: str, job_id: str) -> AsyncGenerator:
        """
        Main method to run the scoring and improving process and return dict.
        """

        yield f"data: {json.dumps({'status': 'starting', 'message': 'Analyzing resume and job description...'})}\n\n"
        await asyncio.sleep(2)

        resume, processed_resume = await self._get_resume(resume_id)
        job, processed_job = await self._get_job(job_id)

        yield f"data: {json.dumps({'status': 'parsing', 'message': 'Parsing resume content...'})}\n\n"
        await asyncio.sleep(2)

        extracted_job_keywords = ", ".join(
            json.loads(processed_job.extracted_keywords).get("extracted_keywords", [])
        )

        extracted_resume_keywords = ", ".join(
            json.loads(processed_resume.extracted_keywords).get(
                "extracted_keywords", []
            )
        )

        resume_embedding = await self.embedding_manager.embed(text=resume.content)
        extracted_job_keywords_embedding = await self.embedding_manager.embed(
            text=extracted_job_keywords
        )

        yield f"data: {json.dumps({'status': 'scoring', 'message': 'Calculating compatibility score...'})}\n\n"
        await asyncio.sleep(3)

        cosine_similarity_score = self.calculate_cosine_similarity(
            extracted_job_keywords_embedding, resume_embedding
        )

        yield f"data: {json.dumps({'status': 'scored', 'score': cosine_similarity_score})}\n\n"

        yield f"data: {json.dumps({'status': 'improving', 'message': 'Generating improvement suggestions...'})}\n\n"
        await asyncio.sleep(3)

        updated_resume, updated_score = await self.improve_score_with_llm(
            resume=resume.content,
            extracted_resume_keywords=extracted_resume_keywords,
            job=job.content,
            extracted_job_keywords=extracted_job_keywords,
            previous_cosine_similarity_score=cosine_similarity_score,
            extracted_job_keywords_embedding=extracted_job_keywords_embedding,
        )

        for i, suggestion in enumerate(updated_resume):
            yield f"data: {json.dumps({'status': 'suggestion', 'index': i, 'text': suggestion})}\n\n"
            await asyncio.sleep(0.2)

        final_result = {
            "resume_id": resume_id,
            "job_id": job_id,
            "original_score": cosine_similarity_score,
            "new_score": updated_score,
            "updated_resume": markdown.markdown(text=updated_resume),
        }

        yield f"data: {json.dumps({'status': 'completed', 'result': final_result})}\n\n"
