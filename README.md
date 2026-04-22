# AI Exam Paper Generator

## Project Overview

An AI-powered exam paper generator that creates MCQs, short questions, and long questions using Groq's LLM API with Bloom's Taxonomy filtering.

## Features

- Generate multiple choice questions with 4 options
- Generate short answer questions (2-3 sentences)
- Generate long answer questions (detailed)
- Complete answer key with explanations
- Bloom's Taxonomy filtering (6 cognitive levels)
- Download exam as text file
- Professional gradient UI

## Technologies Used

- Python 3.8+
- Gradio (Web Interface)
- Groq API (Llama 3.3 Model)
- JSON (Data Format)

## How to Run

```bash
# Install dependencies
pip install gradio groq

# Run the application
python exam_generator.py