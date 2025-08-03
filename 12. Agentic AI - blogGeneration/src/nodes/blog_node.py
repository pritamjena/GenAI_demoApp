from src.states.blogstate import BlogState

class BlogNode:
    """
    A class to represent the blog node
    """

    def __init__(self,llm):
        self.llm=llm

    
    def title_creation(self,state:BlogState):
        """
        create the title for the blog
        """

        if "topic" in state and state["topic"]:
            generate_language = state.get("generate_language", "English")
            
            prompt = f"""
You are a distinguished content strategist and subject matter expert with 15+ years of experience in creating authoritative, research-driven content. Your task is to create ONE exceptional blog title for the topic: "{{topic}}".

IMPORTANT: Generate the title in {generate_language} language. The topic is provided in English, but your title should be in {generate_language}.

CRITICAL REQUIREMENTS:
1. **Authority & Credibility**: Position the content as expert-level, research-backed, and authoritative
2. **Specificity & Precision**: Use precise, technical, or industry-specific terminology that demonstrates expertise
3. **Value-Focused**: Emphasize the depth of knowledge, insights, or comprehensive analysis provided
4. **Professional Tone**: Avoid sensationalism, clickbait, or viral marketing tactics
5. **Intellectual Appeal**: Target readers seeking substantial, educational content
6. **SEO Optimization**: Include relevant keywords naturally while maintaining readability
7. **Length**: Keep between 60-80 characters for optimal professional presentation
8. **Language**: Create the title in {generate_language} language

TITLE STRUCTURE EXAMPLES:
- "Comprehensive Analysis: [Specific Aspect] of [Topic]"
- "[Number] Evidence-Based Strategies for [Specific Outcome]"
- "The Definitive Guide to [Specific Technique/Method]"
- "Research Insights: [Specific Finding] in [Topic Area]"
- "Expert Perspectives on [Specific Challenge] in [Topic]"
- "[Topic]: A Deep Dive into [Specific Component]"

AVOID THESE PATTERNS:
❌ "Shocking Truth About [Topic]"
❌ "[Number] Ways to [Generic Action]"
❌ "You Won't Believe [Topic]"
❌ "The Secret to [Topic]"
❌ "How to [Action] in [Unrealistic Timeframe]"
❌ "Viral [Topic] Tips"

RESPONSE FORMAT:
Return ONLY the title text in {generate_language} language, no additional formatting, quotes, or explanations.

Topic: {{topic}}
Title:"""
            
            system_message = prompt.format(topic=state["topic"])
            # print(system_message)
            response = self.llm.invoke(system_message)
            # print(response)
            return {"blog":{"title":response.content}}
        
    def content_generation(self,state:BlogState):
        if "topic" in state and state["topic"]:
            generate_language = state.get("generate_language", "English")
            
            system_prompt = f"""You are a distinguished subject matter expert and content strategist with extensive experience in creating comprehensive, research-driven blog content. Create an authoritative, in-depth blog post for the topic: "{{topic}}".

IMPORTANT: Generate the entire blog content in {generate_language} language. The topic is provided in English, but your response should be in {generate_language}.

CONTENT REQUIREMENTS:
1. **Research-Backed**: Include relevant statistics, studies, and expert opinions where applicable
2. **Comprehensive Coverage**: Provide thorough analysis covering multiple aspects of the topic
3. **Expert Authority**: Demonstrate deep knowledge and professional expertise
4. **Educational Value**: Focus on teaching and informing rather than entertaining
5. **Practical Application**: Include actionable insights, methodologies, and frameworks
6. **Critical Analysis**: Present balanced perspectives and address potential challenges
7. **Professional Depth**: Aim for 1500-2500 words of substantial, high-quality content
8. **Academic Rigor**: Use proper citations, references, and evidence-based claims
9. **Language**: Write the entire content in {generate_language} language

CONTENT STRUCTURE:
- **Executive Summary**: Brief overview of key insights and takeaways
- **Context & Background**: Establish the topic's relevance and current landscape
- **Core Analysis**: 4-6 comprehensive sections covering different aspects
- **Evidence & Examples**: Include relevant data, case studies, and real-world applications
- **Methodology/Framework**: Provide structured approaches or systematic methods
- **Challenges & Considerations**: Address potential obstacles and limitations
- **Future Implications**: Discuss trends, developments, and forward-looking insights
- **Conclusion**: Synthesize key findings and provide strategic recommendations

FORMATTING REQUIREMENTS:
- Use professional Markdown formatting
- Include H2 and H3 headings for clear structure
- Use bullet points and numbered lists for complex information
- Bold important concepts and key takeaways
- Include code blocks, diagrams, or structured examples where relevant
- Use blockquotes for expert opinions or important citations
- Include reference sections where appropriate

TONE & STYLE:
- Professional and authoritative
- Analytical and evidence-based
- Educational and informative
- Balanced and objective
- Accessible to educated professionals
- Avoid sensationalism or marketing language
- Written in {generate_language} language

Topic: {{topic}}
Blog Content:"""
            system_message = system_prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog": {"title": state['blog']['title'], "content": response.content}}