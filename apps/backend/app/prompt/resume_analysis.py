PROMPT = """
You are an expert resume analysis specialist. Analyze how well the resume matches the job description and provide a detailed analysis in valid JSON format.

Job Description:
{raw_job_description}

Job Keywords: {extracted_job_keywords}

Resume Content:
{raw_resume}

Resume Keywords: {extracted_resume_keywords}

Current Match Score: {current_cosine_similarity:.4f}

Return your analysis as valid JSON with this exact structure:

{{
  "overall_match_score": "75%",
  "strengths": ["List 3-5 specific strengths"],
  "gaps": ["List 3-5 missing requirements"],
  "keyword_analysis": {{
    "matched_keywords": ["list of matched keywords"],
    "missing_keywords": ["list of missing important keywords"]
  }},
  "improvement_recommendations": [
    {{
      "priority": "high",
      "recommendation": "specific suggestion",
      "impact": "expected improvement"
    }}
  ]
}}

IMPORTANT: Return only valid JSON. No explanations before or after.
"""