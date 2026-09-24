"""
LLM Prompts and Templates

Prompt templates for CV/JD generation and validation.

Templates:
- build_cv_v3: Generate realistic CVs from competence profiles
- build_jd_v3: Generate realistic job descriptions from competence profiles
- checker_v2: Validate and refine generated text against required competencies

Variables:
- {JOB_TITLE}: Occupation title
- {COMPETENCE_ITEMS}: Formatted list of required competencies
- {STYLE_REQ}: Writing style requirements
- {PREV_TEXT}: Previously generated text (for validation)
"""

# ============================================================================
# EXAMPLE OUTPUTS
# ============================================================================

json_example = """{
  "human tasks": [
    {
      "type": "task",
      "short name": "design and develop robotic systems",
      "quote": "Designed and developed multiple robotic systems for industrial and commercial applications, including a robotic arm for manufacturing and a self-driving vehicle for transportation."
    },
    {
      "type": "task",
      "short name": "lead engineering team",
      "quote": "Led a team of engineers to implement and test these systems, resulting in significant improvements in efficiency and productivity."
    },
    {
      "type": "task",
      "short name": "conduct research in robotics",
      "quote": "Conducted research in robotics and autonomous systems, focusing on topics such as robotic perception, planning, and control."
    },
    {
      "type": "task",
      "short name": "publish research papers",
      "quote": "Published multiple papers in top-tier conferences and journals, including IEEE Robotics and Automation."
    },
    {
      "type": "task",
      "short name": "present research internationally",
      "quote": "Presented research at international conferences."
    }
  ],
  "human aspects": [
    {
      "type": "skill",
      "short name": "python",
      "quote": "Proficient in programming languages such as Python."
    },
    {
      "type": "skill",
      "short name": "c++",
      "quote": "Proficient in programming languages such as Python, C++, and Java."
    },
    {
      "type": "skill",
      "short name": "java",
      "quote": "Proficient in programming languages such as Python, C++, and Java."
    },
    {
      "type": "skill",
      "short name": "ROS",
      "quote": "Expertise in robotics frameworks including ROS and OpenCV."
    },
    {
      "type": "skill",
      "short name": "OpenCV",
      "quote": "Expertise in robotics frameworks including ROS and OpenCV."
    },
    {
      "type": "skill",
      "short name": "computer vision",
      "quote": "Experience with computer vision, machine learning, and natural language processing."
    },
    {
      "type": "skill",
      "short name": "machine learning",
      "quote": "Experience with computer vision, machine learning, and natural language processing."
    },
    {
      "type": "skill",
      "short name": "natural language processing",
      "quote": "Experience with computer vision, machine learning, and natural language processing."
    },
    {
      "type": "skill",
      "short name": "operating systems",
      "quote": "Proficiency in operating systems including Windows, Linux, and macOS."
    },
    {
      "type": "knowledge",
      "short name": "robotic perception, planning, and control",
      "quote": "Focusing on topics such as robotic perception, planning, and control."
    },
    {
      "type": "knowledge",
      "short name": "robotics design and implementation",
      "quote": "Utilizing my skills and experience in robotics design, development, and implementation."
    },
    {
      "type": "attitude",
      "short name": "pursuit of challenging and rewarding work",
      "quote": "To obtain a challenging and rewarding position as a Senior Robotics Engineer."
    },
    {
      "type": "trait",
      "short name": "innovation-driven",
      "quote": "To contribute to the success of a dynamic and innovative organization."
    }
  ]
}
"""

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

PROMPTS = {
    # Legacy prompts (not currently used)
    "resume generation": """
        You are searching for a job. Write a 
        realistic resume based on fictitious skills,
        knowledge and abilities (i.e. they don't have to correspond with your own). 
        The resume must be pure text (so do not use bullet points or tables, you 
        may use headings and white lines to format the text). The resume is for a person called 
        """,
    "content extraction simple": 
        f"""You are a text analyser. You obtain the following resume. Please provide a structured overview 
        of the contents of this resume in json format. Outline the following classes: knowledge, attitudes, 
        skills and capabilities (these are called items). You may add an additional category for relevant 
        items called 'other' that cannot be categorised as one of the above. For each item you identify you must 
        categorize its type and quote the text from the resume that is relevant to this item. 
        There is a high penalty for missing items. Do not output anything other than json parsable text. 
        Here is an example output {json_example}  Here is the resume:""",
    "content extraction strict": 
        """You are a text analyser. You obtain this resume. Please provide a structured overview 
        of the contents of this resume in json format. Outline the following classes: knowledge, 
        attitudes and capabilities (these are called items). You may add an additional category 
        for relevant items that cannot be categorised as one of the above. For each item you identify 
        you must categorize its type and quote the text from the resume that is relevant to this item. 
        Please try to be as complete as possible. There is a high penalty for missing items""",
    "core-o extraction":
        f"""
        You are a text analyser. You have been given this resume and your task is to extract all relevant information from this resume.
        Please extract the following classes: 'Human tasks' and 'human capabilities'. Human tasks are performed by the
        individual, may use resources and can generate a certain task output or competence outcome. 'Human Aspects' can be 
        human capabilities (skills at a certain profeciency), knowledge, attitudes, human qualities or traits. There is a high penalty
         for missing instances of either class. For each instance, extract a quote which best represent the extracted instance. 
         If the quote is a non-continuous slice of text please split the continuous fragments by '...' in the output. There is high
          a penalty for omitting relevant information in the extracted quote for each instance. Please provide the extracted information in a json 
          format structured in the format {json_example}. DO not output anything other than the JSON string Here is the resume:
        """,
    "core-o extraction v2 cv":
          """
          You are an information extraction model for resumes.

          Your task is to identify and classify competence-relevant concepts and evidence in the text using the ontology classes listed below.

          Allowed classes (use ONLY the pref_label values exactly as written):

          [
            {NER_CLASSES}
          ]

          Extraction rules:
          - Each concept must be classified to exactly one ontology class. 
          - If a concepts may fit multiple classes to your judgement, provide seperate concepts for each.
          - Each concept must be given a short name (1-5 words) that uniquely identifies the concept it represents. e.g. if the concept is "Knowledge of the organization", the short name could be "Organization knowledge".
          - For each concept, provide a exact substring from the input text that is relevant to the concept. Please include all relevant information in the evidence.
          - For each evidence string, provide the start and end character offsets of the substring from the input text.
          - include ALL competence relevant information n the text. There is a penalty for missing concepts.
          - Do NOT invent information. If nothing is relevant, return an empty list.


          Output format (JSON only, no extra text):

          {{
            "entities": [
              {{
                "text": "self defined short name of the concept",
                "kind": "one of the allowed pref_label values exactly",
                "start_char": 0,
                "end_char": 0,
                "evidence": "full evidence text"
              }}
            ]
          }}

          Input text:
          >>>
          {INPUT_TEXT}
          <<<
          """,
      "core-o extraction v2 jd":
          """
          You are an information extraction model for job descriptions.

          Your task is to identify and classify competence-relevant concepts and evidence in the text using the ontology classes listed below.

          Allowed classes (use ONLY the pref_label values exactly as written):

          [
            {NER_CLASSES}
          ]

          Extraction rules:
          - Each concept must be classified to exactly one ontology class. 
          - If a concepts may fit multiple classes to your judgement, provide seperate concepts for each.
          - Each concept must be given a short name (1-5 words) that uniquely identifies the concept it represents. e.g. if the concept is "Knowledge of the organization", the short name could be "Organization knowledge".
          - For each concept, provide a exact substring from the input text that is relevant to the concept. Please include all relevant information in the evidence.
          - Each concept must be given an importance which is either "required" or "nice-to-have"
          - For each evidence string, provide the start and end character offsets of the substring from the input text.
          - include ALL competence relevant information n the text. There is a penalty for missing concepts.
          - Do NOT invent information. If nothing is relevant, return an empty list.


          Output format (JSON only, no extra text):

          {{
            "entities": [
              {{
                "text": "self defined short name of the concept",
                "kind": "one of the allowed pref_label values exactly",
                "start_char": 0,
                "end_char": 0,
                "evidence": "full evidence text",
                "importance": "required" or "nice-to-have"
              }}
            ]
          }}

          Input text:
          >>>
          {INPUT_TEXT}
          <<<
          """,
    "relation extraction":
          """
          You are an information extraction model for resumes and job descriptions.

          Your task is to identify whether the existence of relation {RELATION} is supported by the given evidence.

          Here is the evidence:
          {EVIDENCE}
          
          Extraction rules
          - Please respond ONLY with a confidence score between 0 and 1 a nothing else
          - 0 means the existence of the relationship is definitely not supported by the evidence 
          - 1 means  the existence of the relationship is it is certainly supported by the evidence. 
          - Do NOT provide any explanations or justifications.

          """,
      "concept matching":
          """
          You are a competence concept matching agent. 
          Your task is to determine if two competece-level concepts refer to the same thing.

          Concept 1: {CONCEPT1} 
          Concept 2: {CONCEPT2} 

          Extraction rules
          - Please respond ONLY with a matching score between 0 and 1 a nothing else
          - 0 means the concepts refer to completey different things.
          - 1 means  the concepts refer to exactly the same thing.
          - Do NOT provide any explanations or justifications.

          """,
      "build_cv":
          """
          You are a resume generator.

          Your task is to generate a resume for  a fictitious candidate  applying for the following job title: {JOB_TITLE}

          The resume should describe the following skills: 
          {SKILLS}

          The resume should describe the following abilities: 
          {ABILITIES}

          The resume should describe the following knowledge: 
          {KNOWLEDGE}

          The resume should describe the following tasks: 
          {TASKS}



          Generation rules:
          - Please respond ONLY with a resume text.
          - Do NOT use placeholders for personal information in the resume, put fictisious information instead.
          - Do NOT add any skills, abilities, knowledge or tasks that are not mentioned in the input.
          - Do NOT omit any skills, abilities, knowledge or tasks that are mentioned in the input.
          - you are free to choose the order of the skills, abilities, knowledge and tasks in the resume.
          - your are free to choose the style of the resume
          """,

      "build cv v2":
           """
          You are a resume generator.

          Your task is to generate a resume for  a fictitious candidate  applying for the following job title: {JOB_TITLE}

          The resume should describe the following competence related items: 
          {ITEMS}


          
          Generation rules:
          - Please respond ONLY with a resume text. Do NOT include demographic information or placeholders. 
          - Do NOT add any skills, abilities, knowledge or tasks that are not mentioned in the input.
          - Do NOT omit any skills, abilities, knowledge or tasks that are mentioned in the input.
          - you are free to choose the order of the skills, abilities, knowledge and tasks in the resume.
          - {STYLE_REQ}
          """,

        "build_job_description":
          """
          You are a job description generator.

          Your task is to generate a job description for the following job title: {JOB_TITLE}

          The job description should describe the following skills: 
          {SKILLS}

          The job description should describe the following abilities: 
          {ABILITIES}

          The job description should describe the following knowledge: 
          {KNOWLEDGE}

          The job description should describe the following tasks: 
          {TASKS}



          Generation rules:
          - Please respond ONLY with a job description text.
          - Do NOT add any skills, abilities, knowledge or tasks that are not mentioned in the input.
          - Do NOT omit any skills, abilities, knowledge or tasks that are mentioned in the input.
          - take into acount the importance of the skills, abilities, knowledge and tasks as given in the input.
          - you are free to choose the order of the skills, abilities, knowledge and tasks in the job description.
          - your are free to choose the style the job description
          """,

  "build jd v2":
      """
    You are a job description generator.

    Your task is to generate job description for for the following job: {JOB_TITLE}

    The job description should describe the following competence related items: 
    {ITEMS}
    
    Generation rules:
    - Please respond ONLY with a job decription text. do NOT use any placeholders in the text for demographic information
    - Do NOT add any skills, abilities, knowledge or tasks that are not mentioned in the input.
    - Do NOT omit any skills, abilities, knowledge or tasks that are mentioned in the input.
    - you are free to choose the order of the skills, abilities, knowledge and tasks in the job description.
    - {STYLE_REQ}
    """,
  "check text":
  """
    You are a data validation agent. 
    Your job is to check whether the following text contains certain competence related items: 

    {TEXT}

    The text should contain the following competence related items:

    {COMPETENCE_ITEMS}

    - remove ANY placeholders that are in the text. Do NOT replace them. 
    - If ANY competence items are missing please add them into the text. 
    - If there are any competence related items in the text that are not mentioned in the items please remove them from the text.
    - Please make sure that the importance of items with importance labels are properly reflected in text. Only change the text if this is not already the case.
    - Make NO OTHER CHANGES to the original text. 
    - respond ONLY with the revised text.
  """,

  "build_jd_v3":
"""
You are a job description/vacancy writing agent

GENERATION RULES
- Respond ONLY with the job description text
- Do NOT use placeholders
- Do NOT add competence items not listed
- Do NOT omit any listed competence items
- You may choose wording and ordering
- style requirement: {STYLE_REQ}

IMPORTANCE INTERPRETATION
Competence items are formatted as: "kind - name [importance label]: description"

ONLY items with bracketed importance labels should have importance reflected in your text:
- [extremely important] → central, essential, critical, required, mandatory
- [very important] → strongly preferred, highly valued, very important, highly significant
- [important] → important, notable, meaningful
- [somewhat important] → beneficial, a plus, preferred, nice to have
- [not important] → optional, secondary, not required, not important

Items WITHOUT brackets (e.g., "tech_skill - Microsoft Office: productivity software") have NO importance label.
For these items: simply mention them factually without any importance qualifiers (do not call them "optional", "required", "preferred", etc.)

STRICT CONSTRAINTS
- The description must be fully derivable from the provided competence items
- If uncertain whether something is implied, DO NOT include it
- Do NOT introduce assumptions, tools, technologies, credentials, or responsibilities not explicitly listed

OUTPUT REQUIREMENTS
- Plain prose only
- Do not mention instructions
- Keep the total output under 750 words

EXAMPLE JOB DESCRIPTION
General and Operations Managers

We are seeking an experienced General and Operations Manager to oversee operations and personnel functions within production facilities. The successful candidate will hold a Bachelor's Degree and have over 10 years of related experience. Administration and management knowledge is critical for this role and will be applied daily.

Key responsibilities include managing the movement of goods into and out of production facilities to ensure efficiency, effectiveness, or sustainability of operations, and performing personnel functions such as selection, training, or evaluation. The role requires the ability to instruct and teach others how to do tasks and to understand and act on written work-related documents.

Required knowledge and capabilities include personnel and human resources knowledge (important): knowledge of principles and procedures for personnel recruitment, selection, training, compensation and benefits, labor relations and negotiation, and personnel information systems; and administration and management knowledge (required): knowledge of business and management principles involved in strategic planning, resource allocation, human resources modeling, leadership technique, production methods, and coordination of people and resources.

Skills and abilities include reading comprehension (important): understanding written sentences and paragraphs in work-related documents; instructing (important): teaching others how to do something; memorization (important): the ability to remember information such as words, numbers, pictures, and procedures; and speech clarity (important): the ability to speak clearly so others can understand you.

Work style expectations include a tendency to take initiative and to be proactive in taking on extra responsibilities, and a strong tendency toward integrity, being honest and ethical at work.

Education: Bachelor's Degree required.
Experience: Over 10 years of related experience required.

TASK
Generate a job description for the following job.

JOB TITLE
{JOB_TITLE}

COMPETENCE ITEMS (STRICTLY LIMITED SCOPE)
Each competence item may include an importance label.

>>>
{COMPETENCE_ITEMS}
<<<

""",

  "build_cv_v3":
"""
You are a resume writing agent

GENERATION RULES
- Respond ONLY with the resume text
- Do NOT use placeholders
- Do NOT include any personal information like name, or contact information
- Do NOT add competence items not listed
- Do NOT omit any listed competence items
- You may choose wording and ordering
- style requirement: {STYLE_REQ}

STRICT CONSTRAINTS
- The description must be fully derivable from the provided competence items
- If uncertain whether something is implied, DO NOT include it
- Do NOT introduce assumptions, tools, technologies, credentials, or responsibilities not explicitly listed

OUTPUT REQUIREMENTS
- Plain prose only
- Do not mention instructions
- Keep the total output under 750 words

EXAMPLE RESUME
Profile: Experienced critical care nurse with seven years of direct patient care experience. Educated to the Associate's Degree level. Focused on delivering attentive patient care while applying structured learning approaches and clinical reasoning.

Experience: Over 6 years of hands-on work in critical care settings, performing pulmonary assessments to identify abnormal respiratory patterns or breathing sounds that indicate problems; providing post-mortem care; and administering medications intravenously, by injection, orally, through gastric tubes, or by other methods. Regularly responsible for obtaining and seeing to the appropriate use of equipment, facilities, and materials needed to complete clinical tasks.

Key competencies and skills: Learning Strategies — selecting and using training and instructional methods appropriate to the situation when learning or teaching new procedures and practices. Management of Material Resources — obtaining and ensuring appropriate use of equipment, facilities, and materials required for patient care tasks. Persuasion — persuading others to change their minds or behavior when necessary to support safe care and adherence to clinical plans. Critical Thinking — using logic and reasoning to identify strengths and weaknesses of alternative solutions, conclusions, or approaches to clinical problems.

Abilities: Fluency of Ideas — able to generate a number of ideas on a topic to support problem solving and care planning.

Work style: Empathy — consistently shows concern for others and sensitivity to patients' and families' needs and feelings at work.

Education: Associate's Degree.

TASK
Generate a resume for a candidate applying to the following job. 

JOB TITLE
{JOB_TITLE}

COMPETENCE ITEMS
>>>
{COMPETENCE_ITEMS}
<<<



""",
"build_jd_v4":
"""You are a job description writing agent.

TASK
Write a job description for the role and competence items provided below.

OUTPUT RULES
- Plain prose only — no bullet points, headers, or lists
- Under 750 words
- Do not mention these instructions
- Do not use placeholders
- Include every competence item listed — no more, no less

IMPORTANCE INTERPRETATION
Competence items are formatted as: "kind - name [importance label]: description"

ONLY items with bracketed importance labels should have importance reflected in your text:
- [extremely important] → central, essential, critical, required, mandatory
- [very important] → strongly preferred, highly valued, very important, highly significant
- [important] → important, notable, meaningful
- [somewhat important] → beneficial, a plus, preferred, nice to have
- [not important] → optional, secondary, not required, not important

Items WITHOUT brackets have NO importance label. For these items: simply mention them factually without any importance qualifiers.

STYLE
{STYLE_REQ}

This instruction overrides any implicit style you may infer from the reference example below.
In all cases, ensure the job description is realistically written.
Vary sentence length, paragraph structure, and vocabulary freely.

STRICT CONTENT CONSTRAINTS
- Only include what is explicitly listed in the competence items
- Do not introduce tools, technologies, credentials, or responsibilities not listed
- If uncertain whether something is implied, omit it

---

JOB TITLE
{JOB_TITLE}

COMPETENCE ITEMS
>>>
{COMPETENCE_ITEMS}
<

---

REFERENCE EXAMPLE
The following is one possible style — treat it as a lower bound for quality,
not a template to follow. Your output should differ meaningfully in structure and voice.

General and Operations Managers

We are seeking an experienced General and Operations Manager to oversee operations and personnel functions within production facilities. The successful candidate will hold a Bachelor's Degree and have over 10 years of related experience. Administration and management knowledge is critical for this role and will be applied daily.

Key responsibilities include managing the movement of goods into and out of production facilities to ensure efficiency, effectiveness, or sustainability of operations, and performing personnel functions such as selection, training, or evaluation. The role requires the ability to instruct and teach others how to do tasks and to understand and act on written work-related documents.

Required knowledge and capabilities include personnel and human resources knowledge (important): knowledge of principles and procedures for personnel recruitment, selection, training, compensation and benefits, labor relations and negotiation, and personnel information systems; and administration and management knowledge (required): knowledge of business and management principles involved in strategic planning, resource allocation, human resources modeling, leadership technique, production methods, and coordination of people and resources.

Skills and abilities include reading comprehension (important): understanding written sentences and paragraphs in work-related documents; instructing (important): teaching others how to do something; memorization (important): the ability to remember information such as words, numbers, pictures, and procedures; and speech clarity (important): the ability to speak clearly so others can understand you.

Work style expectations include a tendency to take initiative and to be proactive in taking on extra responsibilities, and a strong tendency toward integrity, being honest and ethical at work.

Education: Bachelor's Degree required.
Experience: Over 10 years of related experience required.

""",
"build_cv_v4":
"""
You are a resume writing agent.

TASK
Write a resume for a candidate the is focused in th role and competence items provided below.

OUTPUT RULES
- Plain prose only — no bullet points, headers, or lists
- Under 750 words
- Do not mention these instructions
- Do not use placeholders
- Do not include any personal information such as name or contact information
- Include every competence item listed — no more, no less

STYLE
{STYLE_REQ}

This instruction overrides any implicit style you may infer from the reference example below. 
In all cases, ensure the resume is realistically written.  
Vary sentence length, paragraph structure, and vocabulary freely.

STRICT CONTENT CONSTRAINTS
- Only include what is explicitly listed in the competence items. 
- You may choose related experience and required education from examples or spans if multiple are listed, but do not introduce any new experience or education that is not listed. In the case of a span, choose the maximum level mentioned.
- Do not introduce tools, technologies, credentials, or responsibilities not listed
- If uncertain whether something is implied, omit it

---

ROLE
{JOB_TITLE}

COMPETENCE ITEMS
>>>
{COMPETENCE_ITEMS}
<

---

REFERENCE EXAMPLE
The following illustrates content completeness and quality — not a template to follow.
Your output must differ meaningfully in structure, voice, and paragraph organization.

Profile: Experienced critical care nurse with seven years of direct patient care experience. Educated to the Associate's Degree level. Focused on delivering attentive patient care while applying structured learning approaches and clinical reasoning.

Experience: Over 6 years of hands-on work in critical care settings, performing pulmonary assessments to identify abnormal respiratory patterns or breathing sounds that indicate problems; providing post-mortem care; and administering medications intravenously, by injection, orally, through gastric tubes, or by other methods. Regularly responsible for obtaining and seeing to the appropriate use of equipment, facilities, and materials needed to complete clinical tasks.

Key competencies and skills: Learning Strategies — selecting and using training and instructional methods appropriate to the situation when learning or teaching new procedures and practices. Management of Material Resources — obtaining and ensuring appropriate use of equipment, facilities, and materials required for patient care tasks. Persuasion — persuading others to change their minds or behavior when necessary to support safe care and adherence to clinical plans. Critical Thinking — using logic and reasoning to identify strengths and weaknesses of alternative solutions, conclusions, or approaches to clinical problems.

Abilities: Fluency of Ideas — able to generate a number of ideas on a topic to support problem solving and care planning.

Work style: Empathy — consistently shows concern for others and sensitivity to patients' and families' needs and feelings at work.

Education: Associate's Degree.
""",
"checker_v2":
"""
You are a controlled data validation system.

VALIDATION PROCEDURE
1. Identify competence items present in the text
2. Identify missing competence items
3. Identify competence items not listed
4. Identify any Personally Identifyable Information (PII) including names, pronouns, contact details, and identifying references.
5. Apply minimal edits required to enforce compliance


EDITING RULES
- Remove placeholders
- Remove any PII and rewrite using neutral, non-identifying language when necessary
- Add missing competence items
- Remove competence items not listed
- Ensure that the importance of items with importance labels are properly reflected in text 
- Remove importance indicators for items that are listed without importance labels


PLACEHOLDER DEFINITION
A placeholder is any token such as:
[...], <...>, XXX, ABC, XYZ, TBD, PLACEHOLDER

IMPORTANCE INTERPRETATION
Competence items are formatted as: "kind - name [importance label]: description"

ONLY items with bracketed importance labels should have importance reflected in the text:
- [extremely important] → central, essential, critical, required, mandatory
- [very important] → strongly preferred, highly valued, very important, highly significant
- [important] → important, notable, meaningful
- [somewhat important] → beneficial, a plus, preferred, nice to have
- [not important] → optional, secondary, not required, not important

Items WITHOUT brackets (e.g., "tech_skill - Microsoft Office: productivity software") have NO importance label.
For these items: mention them factually without any importance qualifiers (do not call them "optional", "required", "preferred", etc.) 

EDITING CONSTRAINTS
- Preserve original wording whenever possible
- Only modify sentences directly affected by violations
- Minimize edits
- Do NOT improve style, tone, grammar, or fluency unless required by rule violations
- Maintain grammatical correctness
- Make no other changes

OUTPUT REQUIREMENTS
- Respond ONLY with the revised text
- Keep the total output under 750 words

TEXT TO VALIDATE
<<<
{TEXT}
>>>

REQUIRED COMPETENCE ITEMS
<<<
{COMPETENCE_ITEMS}
>>>

"""
}   

def prompt_builder(prompt_template: str, args: dict) -> str:
    prompt = prompt_template.format(**args)
    return prompt

if __name__ == "__main__":  
    print("build prompt")