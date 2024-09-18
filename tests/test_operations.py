import pytest
import pandas as pd
from src.pipeline.predict_pipeline import PredicPipeline, CustomData
from src.pipeline.train_pipeline import TrainPipeline

@pytest.mark.training
def test_train_pipeline():
    """
    Test the training pipeline to ensure that the data ingestion, transformation,
    and model training steps are functioning correctly.
    """
    try:
        # Initialize and run the training pipeline
        pipeline = TrainPipeline()
        pipeline.start_training_pipeline()
        
        # If no exceptions occur, pass the test
        assert True, "Train pipeline ran successfully"
    except Exception as e:
        # Fail the test if an error occurs
        pytest.fail(f"Train pipeline failed with error: {e}")


@pytest.mark.prediction
def test_predict_pipeline():
    """
    Test the prediction pipeline to ensure that it can handle input data and make predictions.
    """
    try:
        # Create a sample input similar to what the model expects
        custom_data = CustomData(
            gender="female",
            race_ethnicity="group B",
            parental_level_of_education="bachelor's degree",
            lunch="standard",
            test_preparation_course="none",
            reading_score=72,
            writing_score=74
        )
        
        # Convert the input to a DataFrame as expected by the predict method
        data_frame = custom_data.get_data_as_data_frame()

        # Initialize the prediction pipeline and make a prediction
        predict_pipeline = PredicPipeline()
        preds = predict_pipeline.predict(data_frame)

        # Check that predictions are returned and are not empty
        assert preds is not None, "Prediction should not be None"
        assert len(preds) > 0, "Prediction should return at least one value"
    except Exception as e:
        # Fail the test if an error occurs
        pytest.fail(f"Prediction pipeline failed with error: {e}")


@pytest.mark.integration
def test_pipeline_integration():
    """
    Test the integration of both the training and prediction pipelines.
    Ensure the end-to-end process works without errors.
    """
    try:
        # Train the model
        train_pipeline = TrainPipeline()
        train_pipeline.start_training_pipeline()

        # Create sample input for prediction
        custom_data = CustomData(
            gender="male",
            race_ethnicity="group C",
            parental_level_of_education="some college",
            lunch="free/reduced",
            test_preparation_course="completed",
            reading_score=80,
            writing_score=78
        )
        
        data_frame = custom_data.get_data_as_data_frame()

        # Predict using the trained model
        predict_pipeline = PredicPipeline()
        preds = predict_pipeline.predict(data_frame)

        # Validate that predictions are returned and not empty
        assert preds is not None, "Prediction should not be None after training"
        assert len(preds) > 0, "Prediction should return at least one value after training"
    except Exception as e:
        pytest.fail(f"Pipeline integration test failed with error: {e}")
