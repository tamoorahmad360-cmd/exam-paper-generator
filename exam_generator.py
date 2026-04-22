"""
AI Exam Paper Generator


Description: This application generates professional exam papers using Groq AI API.
Features include MCQ generation, Bloom's Taxonomy filtering, and answer keys.
"""

import gradio as gr
from groq import Groq
import json
from datetime import datetime
import re
from typing import Dict, List, Tuple

# Global client (will be set when user enters API key)
client = None

def set_api_key(api_key: str):
    """Set the Groq API key"""
    global client
    try:
        client = Groq(api_key=api_key)
        return True, "✅ API Key set successfully!"
    except Exception as e:
        return False, f"❌ Invalid API Key: {str(e)}"

def generate_exam_prompt(topic, num_mcq, num_short, num_long, difficulty):
    """Create a structured prompt for the LLM"""
    
    prompt = f"""Generate a comprehensive exam paper on the topic: "{topic}"

Difficulty Level: {difficulty}

Requirements:
1. {num_mcq} Multiple Choice Questions - each with 4 options (A, B, C, D) and mark correct answer
2. {num_short} Short Answer Questions - expecting 2-3 sentence answers
3. {num_long} Long Answer Questions - expecting detailed paragraph answers
4. Complete Answer Key for ALL questions
5. Bloom's Taxonomy level for EACH question (choose from: Remember, Understand, Apply, Analyze, Evaluate, Create)

IMPORTANT: You MUST respond with ONLY valid JSON in this exact structure:

{{
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "generated_date": "{datetime.now().strftime('%Y-%m-%d')}",
    "mcqs": [
        {{
            "question": "What is the capital of France?",
            "options": ["A) London", "B) Berlin", "C) Paris", "D) Madrid"],
            "correct_answer": "C",
            "bloom_level": "Remember"
        }}
    ],
    "short_questions": [
        {{
            "question": "Explain the process of photosynthesis.",
            "answer_key": "Photosynthesis is the process where plants convert sunlight into energy...",
            "bloom_level": "Understand"
        }}
    ],
    "long_questions": [
        {{
            "question": "Analyze the causes and effects of climate change.",
            "answer_key": "Climate change is caused by greenhouse gas emissions... [detailed answer]",
            "bloom_level": "Analyze"
        }}
    ]
}}

Generate high-quality, accurate, educational content appropriate for {difficulty} level students."""
    
    return prompt

def call_groq_api(prompt: str) -> Tuple[Dict, str]:
    """Call Groq API and return parsed JSON response"""
    if client is None:
        return None, "❌ Please enter your API key first"
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert educator and exam designer. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        response_text = completion.choices[0].message.content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        exam_data = json.loads(response_text)
        return exam_data, None
        
    except json.JSONDecodeError as e:
        return None, f"❌ JSON Parsing Error: {str(e)}\n\nRaw Response:\n{response_text}"
    except Exception as e:
        return None, f"❌ API Error: {str(e)}"

def format_mcqs(mcqs: List[Dict]) -> str:
    """Format MCQs - STRAIGHT ALIGNMENT"""
    if not mcqs:
        return "<p style='color: #666; font-style: italic;'>✨ No MCQs generated</p>"
    
    output = ""
    for i, mcq in enumerate(mcqs, 1):
        output += f"""
<div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 15px 0; border: 1px solid #e2e8f0;'>
    <p style='font-size: 1.05em; font-weight: 600; color: #2c3e50; margin: 0 0 15px 0;'>{i}. {mcq.get('question', 'N/A')}</p>
"""
        for option in mcq.get('options', []):
            output += f"    <p style='margin: 8px 0; color: #4a5568;'>{option}</p>\n"
        output += f"""
</div>
"""
    return output

def format_short_questions(short_qs: List[Dict]) -> str:
    """Format short questions"""
    if not short_qs:
        return "<p style='color: #666; font-style: italic;'>✨ No short questions generated</p>"
    
    output = ""
    for i, q in enumerate(short_qs, 1):
        output += f"""
<div style='background: #fff5f7; padding: 18px; border-radius: 10px; margin: 15px 0; border: 1px solid #fbb6ce;'>
    <p style='font-size: 1.05em; font-weight: 600; color: #2c3e50; margin: 0;'>{i}. {q.get('question', 'N/A')}</p>
</div>
"""
    return output

def format_long_questions(long_qs: List[Dict]) -> str:
    """Format long questions"""
    if not long_qs:
        return "<p style='color: #666; font-style: italic;'>✨ No long questions generated</p>"
    
    output = ""
    for i, q in enumerate(long_qs, 1):
        output += f"""
<div style='background: #f0f9ff; padding: 20px; border-radius: 10px; margin: 15px 0; border: 1px solid #bae6fd;'>
    <p style='font-size: 1.05em; font-weight: 600; color: #2c3e50; margin: 0;'>{i}. {q.get('question', 'N/A')}</p>
</div>
"""
    return output

def filter_by_bloom_level(questions: List[Dict], bloom_level: str) -> List[Dict]:
    """Filter questions by Bloom's Taxonomy level"""
    if bloom_level == "All Levels":
        return questions
    return [q for q in questions if q.get('bloom_level', '') == bloom_level]

def format_complete_exam_with_filter(topic: str, difficulty: str, mcqs: List[Dict], short_qs: List[Dict], long_qs: List[Dict], bloom_level: str) -> str:
    """Format complete exam paper with Bloom filter"""
    
    filtered_mcqs = filter_by_bloom_level(mcqs, bloom_level)
    filtered_short = filter_by_bloom_level(short_qs, bloom_level)
    filtered_long = filter_by_bloom_level(long_qs, bloom_level)
    
    header = f"""
<div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 30px; color: white;'>
    <h1 style='color: white; margin: 0; font-size: 2.2em;'>📚 EXAMINATION PAPER</h1>
    <h2 style='color: #f3f4f6; margin: 15px 0 10px 0;'>{topic}</h2>
    <p style='margin: 5px 0;'>⭐ Difficulty: {difficulty} | 📅 Date: {datetime.now().strftime('%Y-%m-%d')}</p>
    <p style='margin: 10px 0 0 0; background: rgba(255,255,255,0.2); display: inline-block; padding: 5px 15px; border-radius: 20px;'>
        🎯 Bloom's Level: {bloom_level} ({len(filtered_mcqs) + len(filtered_short) + len(filtered_long)} questions)
    </p>
</div>
"""
    
    section_a = f"""
<div style='margin: 30px 0 20px 0;'>
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px 25px; border-radius: 10px; display: inline-block;'>
        <h2 style='color: white; margin: 0;'>📝 SECTION A: Multiple Choice Questions</h2>
    </div>
    <p style='color: #666; margin-top: 10px;'>Select the best answer for each question. Each question carries 1 mark.</p>
    <p style='color: #059669; font-size: 0.9em;'>Showing {len(filtered_mcqs)} of {len(mcqs)} questions</p>
</div>
"""
    
    section_b = f"""
<div style='margin: 40px 0 20px 0;'>
    <div style='background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%); padding: 12px 25px; border-radius: 10px; display: inline-block;'>
        <h2 style='color: white; margin: 0;'>✏️ SECTION B: Short Answer Questions</h2>
    </div>
    <p style='color: #666; margin-top: 10px;'>Answer each question in 2-3 sentences. Each question carries 3 marks.</p>
    <p style='color: #059669; font-size: 0.9em;'>Showing {len(filtered_short)} of {len(short_qs)} questions</p>
</div>
"""
    
    section_c = f"""
<div style='margin: 40px 0 20px 0;'>
    <div style='background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%); padding: 12px 25px; border-radius: 10px; display: inline-block;'>
        <h2 style='color: white; margin: 0;'>📖 SECTION C: Long Answer Questions</h2>
    </div>
    <p style='color: #666; margin-top: 10px;'>Provide detailed explanations. Each question carries 5 marks.</p>
    <p style='color: #059669; font-size: 0.9em;'>Showing {len(filtered_long)} of {len(long_qs)} questions</p>
</div>
"""
    
    mcqs_html = ""
    for i, mcq in enumerate(filtered_mcqs, 1):
        mcqs_html += f"""
<div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 15px 0; border: 1px solid #e2e8f0;'>
    <p style='font-size: 1.05em; font-weight: 600; color: #2c3e50; margin: 0 0 15px 0;'>{i}. {mcq.get('question', 'N/A')}</p>
"""
        for option in mcq.get('options', []):
            mcqs_html += f"    <p style='margin: 8px 0; color: #4a5568;'>{option}</p>\n"
        mcqs_html += f"""
</div>
"""
    
    short_html = ""
    for i, q in enumerate(filtered_short, 1):
        short_html += f"""
<div style='background: #fff5f7; padding: 18px; border-radius: 10px; margin: 15px 0; border: 1px solid #fbb6ce;'>
    <p style='font-size: 1.05em; font-weight: 600; color: #2c3e50; margin: 0;'>{i}. {q.get('question', 'N/A')}</p>
</div>
"""
    
    long_html = ""
    for i, q in enumerate(filtered_long, 1):
        long_html += f"""
<div style='background: #f0f9ff; padding: 20px; border-radius: 10px; margin: 15px 0; border: 1px solid #bae6fd;'>
    <p style='font-size: 1.05em; font-weight: 600; color: #2c3e50; margin: 0;'>{i}. {q.get('question', 'N/A')}</p>
</div>
"""
    
    if not filtered_mcqs and not filtered_short and not filtered_long:
        no_questions_msg = f"""
<div style='text-align: center; padding: 50px; background: #fee2e2; border-radius: 10px; margin: 30px 0;'>
    <p style='font-size: 1.2em; color: #991b1b;'>⚠️ No questions found at {bloom_level} level</p>
    <p style='margin-top: 10px;'>Please select a different level or generate a new exam paper.</p>
</div>
"""
        return header + no_questions_msg
    
    return header + section_a + mcqs_html + section_b + short_html + section_c + long_html

def format_answer_key_filtered(exam_data: Dict, bloom_level: str) -> str:
    """Format answer key for filtered questions"""
    import re
    
    filtered_mcqs = filter_by_bloom_level(exam_data.get('mcqs', []), bloom_level)
    filtered_short = filter_by_bloom_level(exam_data.get('short_questions', []), bloom_level)
    filtered_long = filter_by_bloom_level(exam_data.get('long_questions', []), bloom_level)
    
    output = "\n" + "=" * 60 + "\n"
    output += f"🔑 ANSWER KEY (Filter: {bloom_level})\n"
    output += "=" * 60 + "\n\n"
    
    if filtered_mcqs:
        output += "📝 SECTION A: Multiple Choice Questions\n"
        output += "-" * 40 + "\n"
        for i, mcq in enumerate(filtered_mcqs, 1):
            output += f"{i}. {mcq.get('correct_answer', 'N/A')}\n"
        output += "\n"
    
    if filtered_short:
        output += "✏️ SECTION B: Short Answer Questions\n"
        output += "-" * 40 + "\n"
        for i, q in enumerate(filtered_short, 1):
            answer = q.get('answer_key', 'N/A')
            clean_answer = re.sub(r'<[^>]+>', '', answer)
            clean_answer = ' '.join(clean_answer.split())
            output += f"{i}. {clean_answer}\n\n"
    
    if filtered_long:
        output += "📖 SECTION C: Long Answer Questions\n"
        output += "-" * 40 + "\n"
        for i, q in enumerate(filtered_long, 1):
            answer = q.get('answer_key', 'N/A')
            clean_answer = re.sub(r'<[^>]+>', '', answer)
            clean_answer = ' '.join(clean_answer.split())
            output += f"{i}. {clean_answer}\n\n"
    
    output += "\n" + "=" * 60 + "\n"
    
    if not filtered_mcqs and not filtered_short and not filtered_long:
        output = f"\n⚠️ No questions found at {bloom_level} level.\n"
    
    return output

def generate_exam_paper(topic: str, num_mcq: int, num_short: int, num_long: int, difficulty: str, api_key: str) -> Tuple[str, str, str, str, str, Dict]:
    """Main function to generate exam paper"""
    
    if not api_key:
        return "❌ Please enter your Groq API key", "", "", "", "", None
    
    success, message = set_api_key(api_key)
    if not success:
        return message, "", "", "", "", None
    
    if not topic or topic.strip() == "":
        return "❌ Please enter a topic", "", "", "", "", None
    
    prompt = generate_exam_prompt(topic, num_mcq, num_short, num_long, difficulty)
    exam_data, error = call_groq_api(prompt)
    
    if error:
        return error, "", "", "", "", None
    
    mcqs_section = format_mcqs(exam_data.get('mcqs', []))
    short_section = format_short_questions(exam_data.get('short_questions', []))
    long_section = format_long_questions(exam_data.get('long_questions', []))
    answer_section = format_answer_key_filtered(exam_data, "All Levels")
    complete_exam = format_complete_exam_with_filter(topic, difficulty, exam_data.get('mcqs', []), exam_data.get('short_questions', []), exam_data.get('long_questions', []), "All Levels")
    
    return complete_exam, mcqs_section, short_section, long_section, answer_section, exam_data

# Custom CSS
custom_css = """
body, .gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    font-family: 'Segoe UI', 'Inter', 'Poppins', sans-serif;
}

.title-clear {
    font-size: 3.5em;
    font-weight: 900;
    margin: 0;
    color: white;
    text-align: center;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
}

.subtitle-clear {
    font-size: 1.3em;
    font-weight: 600;
    color: white;
    text-align: center;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
}

.card {
    background: white;
    border-radius: 15px;
    padding: 25px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.gr-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 12px;
    color: white;
    padding: 14px 35px;
    font-weight: bold;
    font-size: 16px;
    transition: all 0.3s ease;
    cursor: pointer;
}

.gr-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.download-btn {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

.output-box {
    background: white;
    border-radius: 12px;
    padding: 25px;
    margin: 10px 0;
    border: 1px solid #e2e8f0;
}

.bloom-filter {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid #06b6d4;
}

.tabs {
    border-radius: 15px;
    overflow: hidden;
    margin-top: 20px;
}
"""

def create_stunning_interface():
    with gr.Blocks(css=custom_css, title="🎓 AI Exam Paper Generator") as demo:
        
        gr.HTML("""
            <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 30px;">
                <h1 class="title-clear">🎓 AI Exam Paper Generator</h1>
                <p class="subtitle-clear">Powered by Groq AI | Bloom's Taxonomy Filtering | Professional Grade</p>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1, elem_classes="card"):
                gr.Markdown("### 📋 Exam Configuration")
                
                api_key_input = gr.Textbox(
                    label="🔑 Groq API Key",
                    placeholder="Enter your Groq API key here (get from console.groq.com)",
                    type="password",
                    lines=1
                )
                
                topic = gr.Textbox(
                    label="📖 Topic / Subject",
                    placeholder="Example: Python Programming, World War II, Cellular Biology, Photosynthesis...",
                    lines=2
                )
                
                gr.Markdown("---")
                gr.Markdown("### 📊 Question Distribution")
                
                num_mcq = gr.Slider(1, 20, value=5, step=1, label="📝 Multiple Choice Questions")
                num_short = gr.Slider(1, 15, value=5, step=1, label="✏️ Short Answer Questions")
                num_long = gr.Slider(1, 10, value=3, step=1, label="📖 Long Answer Questions")
                
                gr.Markdown("---")
                gr.Markdown("### 🎯 Difficulty Level")
                
                difficulty = gr.Dropdown(
                    choices=["Beginner", "Intermediate", "Advanced", "Expert"],
                    value="Intermediate",
                    label="Select Difficulty"
                )
                
                generate_btn = gr.Button("🚀 Generate Exam Paper", variant="primary", size="lg")
            
            with gr.Column(scale=1, elem_classes="card"):
                gr.Markdown("### 🎯 Bloom's Taxonomy Filter")
                gr.Markdown("""
                <div class="bloom-filter">
                    <p style="margin: 0 0 10px 0;"><strong>Filter questions by cognitive level:</strong></p>
                    <p style="margin: 0; font-size: 0.9em; color: #059669;">✨ This filters ALL tabs including Complete Exam</p>
                </div>
                """)
                
                bloom_filter = gr.Dropdown(
                    choices=["All Levels", "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
                    value="All Levels",
                    label="🧠 Select Bloom's Level"
                )
                
                gr.Markdown("---")
                gr.Markdown("### 💾 Download Options")
                
                download_btn = gr.Button("📥 Download Exam as Text File", variant="secondary", size="lg", elem_classes="download-btn")
                download_file = gr.File(label="📁 Save your exam", visible=True)
        
        with gr.Tabs(elem_classes="tabs"):
            with gr.TabItem("📋 Complete Exam Paper"):
                exam_output = gr.Markdown(elem_classes="output-box")
            with gr.TabItem("📝 MCQs"):
                mcq_output = gr.Markdown(elem_classes="output-box")
            with gr.TabItem("✏️ Short Questions"):
                short_output = gr.Markdown(elem_classes="output-box")
            with gr.TabItem("📖 Long Questions"):
                long_output = gr.Markdown(elem_classes="output-box")
            with gr.TabItem("🔑 Answer Key"):
                answer_output = gr.Markdown(elem_classes="output-box")
        
        status_text = gr.Markdown("")
        
        stored_raw_mcqs = gr.State([])
        stored_raw_short = gr.State([])
        stored_raw_long = gr.State([])
        stored_topic = gr.State("")
        stored_difficulty = gr.State("")
        stored_raw_data = gr.State(None)
        
        def generate_and_display(topic, num_mcq, num_short, num_long, difficulty, api_key):
            complete, mcqs, short_qs, long_qs, answer_key, raw_data = generate_exam_paper(
                topic, int(num_mcq), int(num_short), int(num_long), difficulty, api_key
            )
            
            if "❌" in complete:
                return complete, "", "", "", "", [], [], [], topic, difficulty, None, "❌ Generation failed"
            
            raw_mcqs = raw_data.get('mcqs', [])
            raw_short = raw_data.get('short_questions', [])
            raw_long = raw_data.get('long_questions', [])
            
            return complete, mcqs, short_qs, long_qs, answer_key, raw_mcqs, raw_short, raw_long, topic, difficulty, raw_data, "✅ Exam generated! Use Bloom's filter above."
        
        def apply_filter_to_all(bloom_level, raw_mcqs, raw_short, raw_long, topic, difficulty, raw_data):
            if not raw_mcqs:
                return "⚠️ Generate an exam first", "⚠️ Generate an exam first", "⚠️ Generate an exam first", "⚠️ Generate an exam first"
            
            filtered_complete = format_complete_exam_with_filter(topic, difficulty, raw_mcqs, raw_short, raw_long, bloom_level)
            filtered_mcqs = format_mcqs(filter_by_bloom_level(raw_mcqs, bloom_level))
            filtered_short = format_short_questions(filter_by_bloom_level(raw_short, bloom_level))
            filtered_long = format_long_questions(filter_by_bloom_level(raw_long, bloom_level))
            filtered_answer = format_answer_key_filtered(raw_data, bloom_level)
            
            return filtered_complete, filtered_mcqs, filtered_short, filtered_long, filtered_answer
        
        def create_download_file(topic, exam_content, answer_content):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Exam_{topic.replace(' ', '_')}_{timestamp}.txt"
            
            clean_exam = re.sub(r'<[^>]+>', '', exam_content)
            clean_answer = re.sub(r'<[^>]+>', '', answer_content)
            
            file_content = f"""
{'='*60}
AI EXAM PAPER GENERATOR
{'='*60}
Topic: {topic}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{clean_exam}

{'='*60}
ANSWER KEY
{'='*60}
{clean_answer}
"""
            return filename, file_content
        
        def create_download(topic, exam_content, answer_content):
            if not exam_content or "❌" in exam_content:
                return None
            filename, content = create_download_file(topic, exam_content, answer_content)
            return gr.File(value=content, visible=True, label=f"📁 {filename}")
        
        generate_btn.click(
            generate_and_display,
            inputs=[topic, num_mcq, num_short, num_long, difficulty, api_key_input],
            outputs=[exam_output, mcq_output, short_output, long_output, answer_output, 
                    stored_raw_mcqs, stored_raw_short, stored_raw_long, stored_topic, stored_difficulty, stored_raw_data, status_text]
        )
        
        bloom_filter.change(
            apply_filter_to_all,
            inputs=[bloom_filter, stored_raw_mcqs, stored_raw_short, stored_raw_long, stored_topic, stored_difficulty, stored_raw_data],
            outputs=[exam_output, mcq_output, short_output, long_output, answer_output]
        )
        
        download_btn.click(
            create_download,
            inputs=[stored_topic, exam_output, answer_output],
            outputs=[download_file]
        )
        
        gr.HTML("""
            <div style="text-align: center; padding: 25px; margin-top: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
                <p style="margin: 0;">🎓 AI Exam Paper Generator | Powered by Groq LLM API</p>
                <p style="font-size: 0.9em; margin-top: 8px;">✨ Enter your API key and topic to generate professional exam papers ✨</p>
            </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_stunning_interface()
    demo.launch(share=True)