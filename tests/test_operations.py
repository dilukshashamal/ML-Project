import unittest
import pandas as pd
from src.pipeline.predict_pipeline import PredicPipeline, CustomData
from src.pipeline.train_pipeline import TrainPipeline

class TestOperations(unittest.TestCase):
    
    def test_train_pipeline(self):
        """
        Test the training pipeline to ensure data ingestion, transformation, and model training works correctly.
        """
        try:
            pipeline = TrainPipeline()
            pipeline.start_training_pipeline()
            self.assertTrue(True, "Train pipeline ran successfully")
        except Exception as e:
            self.fail(f"Train pipeline failed with error: {e}")

    def test_predict_pipeline(self):
        """
        Test the predict pipeline to ensure that it can correctly handle predictions.
        """
        try:
            # Creating a sample input similar to what the model expects
            custom_data = CustomData(
                gender="female",
                race_ethnicity="group B",
                parental_level_of_education="bachelor's degree",
                lunch="standard",
                test_preparation_course="none",
                reading_score=72,
                writing_score=74
            )
            
            data_frame = custom_data.get_data_as_data_frame()

            # Initialize the prediction pipeline and predict
            predict_pipeline = PredicPipeline()
            preds = predict_pipeline.predict(data_frame)

            # Check if the prediction has been made successfully
            self.assertIsNotNone(preds, "Prediction should not be None")
            self.assertTrue(len(preds) > 0, "Prediction should return at least one value")
        
        except Exception as e:
            self.fail(f"Prediction pipeline failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
