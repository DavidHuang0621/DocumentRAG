import argparse
import os
from utils import get_doc_tools
from question_handler import read_questions#, process_questions
import pandas as pd
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from helper import load_env

def main():
    load_env()
    parser = argparse.ArgumentParser(description="Query a PDF document with predefined questions.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    # parser.add_argument("questions_path", help="Path to the file containing questions")
    parser.add_argument("output_path", help="Path to save the output CSV")
    args = parser.parse_args()

    # Ensure the OPENAI_API_KEY is set
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
    llm = OpenAI(model="gpt-4-turbo", temperature=0)

    # Get document tools
    vector_tool, summary_tool = get_doc_tools(args.pdf_path, "query_tool")

    # Read questions
    questions = read_questions("./questions.txt")

    agent_worker = FunctionCallingAgentWorker.from_tools(
        [vector_tool, summary_tool], 
        llm=llm, 
        verbose=True
    )
    agent = AgentRunner(agent_worker)

    responses = []
    for question in questions:
        response = agent.query(question)
        responses.append({
            "Question": question,
            "Response": str(response).replace('assistant:','')  # Ensure the response is a string
        })

    # Create a DataFrame from the responses
    df = pd.DataFrame(responses)

    # Save to CSV
    df.to_csv(args.output_path, index=False)
    print(f"Output saved to {args.output_path}")

if __name__ == "__main__":
    main()