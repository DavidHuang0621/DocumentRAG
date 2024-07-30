def read_questions(file_path: str) -> list:
    """Read questions from a file."""
    with open(file_path, 'r') as file:
        questions = [line.strip() for line in file if line.strip()]
    return questions

# def process_questions(questions: list, vector_tool, summary_tool) -> list:
#     """Process each question and return the responses."""
#     responses = []
#     for question in questions:
#         # Determine which tool to use based on the question
#         if "summary" in question.lower():
#             response = summary_tool(question)
#         else:
#             response = vector_tool(question)
        
#         responses.append({
#             "Question": question,
#             "Response": str(response)  # Ensure the response is a string
#         })
#     return responses