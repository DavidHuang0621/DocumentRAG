import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import pandas as pd
from query_tool import main
from utils import get_doc_tools
from question_handler import read_questions, process_questions

class TestQueryTool(unittest.TestCase):

    def setUp(self):
        # Create temporary files for testing
        self.temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        self.temp_questions = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False)
        self.temp_output = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)

        # Write some test questions
        self.test_questions = ["What is X?", "Summarize Y.", "How does Z work?"]
        self.temp_questions.write("\n".join(self.test_questions))
        self.temp_questions.flush()

    def tearDown(self):
        # Clean up temporary files
        os.unlink(self.temp_pdf.name)
        os.unlink(self.temp_questions.name)
        os.unlink(self.temp_output.name)

    @patch('query_tool.get_doc_tools')
    @patch('query_tool.read_questions')
    @patch('query_tool.process_questions')
    def test_main(self, mock_process, mock_read, mock_get_tools):
        # Mock the functions
        mock_get_tools.return_value = (MagicMock(), MagicMock())
        mock_read.return_value = self.test_questions
        mock_process.return_value = [
            {"Question": q, "Response": f"Answer to {q}"} for q in self.test_questions
        ]

        # Set up command-line arguments
        with patch('sys.argv', ['query_tool.py', self.temp_pdf.name, self.temp_questions.name, self.temp_output.name]):
            main()

        # Check if the output file was created and contains the expected data
        self.assertTrue(os.path.exists(self.temp_output.name))
        df = pd.read_csv(self.temp_output.name)
        self.assertEqual(len(df), len(self.test_questions))
        self.assertListEqual(list(df.columns), ['Question', 'Response'])

    def test_read_questions(self):
        questions = read_questions(self.temp_questions.name)
        self.assertListEqual(questions, self.test_questions)

    @patch('utils.SimpleDirectoryReader')
    @patch('utils.VectorStoreIndex')
    @patch('utils.SummaryIndex')
    def test_get_doc_tools(self, mock_summary, mock_vector, mock_reader):
        # Mock the necessary classes and methods
        mock_reader.return_value.load_data.return_value = [MagicMock()]
        mock_vector.return_value.as_query_engine.return_value = MagicMock()
        mock_summary.return_value.as_query_engine.return_value = MagicMock()

        vector_tool, summary_tool = get_doc_tools(self.temp_pdf.name, "test")

        self.assertIsNotNone(vector_tool)
        self.assertIsNotNone(summary_tool)

    def test_process_questions(self):
        # Create mock tools
        mock_vector_tool = MagicMock(return_value="Vector answer")
        mock_summary_tool = MagicMock(return_value="Summary answer")

        responses = process_questions(self.test_questions, mock_vector_tool, mock_summary_tool)

        self.assertEqual(len(responses), len(self.test_questions))
        self.assertEqual(responses[0]['Response'], "Vector answer")
        self.assertEqual(responses[1]['Response'], "Summary answer")
        self.assertEqual(responses[2]['Response'], "Vector answer")

if __name__ == '__main__':
    unittest.main()