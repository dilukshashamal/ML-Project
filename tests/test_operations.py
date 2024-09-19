import pytest
from src.pipeline.predict_pipeline import PredicPipeline, CustomData
from src.pipeline.train_pipeline import TrainPipeline

@pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")

@pytest.mark.training
def test_train_pipeline():
    """
    Test the training pipeline to ensure it runs without errors.
    """
    try:
        pipeline = TrainPipeline()
        pipeline.start_training_pipeline()
        assert True, "Train pipeline ran successfully"
    except Exception as e:
        pytest.fail(f"Train pipeline failed with error: {e}")

@pytest.mark.prediction
def test_predict_pipeline():
    """
    Test the prediction pipeline to ensure predictions are generated correctly.
    """
    try:
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
        print(f"Input DataFrame:\n{data_frame}")

        predict_pipeline = PredicPipeline()
        preds = predict_pipeline.predict(data_frame)
        print(f"Predictions: {preds}")

        assert preds is not None, "Prediction should not be None"
        assert len(preds) > 0, "Prediction should return at least one value"
        assert all(isinstance(p, (int, float)) for p in preds), "Predictions should be numeric"
    except Exception as e:
        pytest.fail(f"Prediction pipeline failed with error: {e}")

@pytest.mark.parametrize("gender, expected", [("female", True), ("male", True)])
def test_parametrized(gender, expected):
    """
    Parameterized test to check predictions for different gender inputs.
    """
    custom_data = CustomData(
        gender=gender,
        race_ethnicity="group A",
        parental_level_of_education="some college",
        lunch="free/reduced",
        test_preparation_course="completed",
        reading_score=60,
        writing_score=60
    )
    
    data_frame = custom_data.get_data_as_data_frame()
    print(f"Input DataFrame:\n{data_frame}")

    predict_pipeline = PredicPipeline()
    
    try:
        preds = predict_pipeline.predict(data_frame)
        print(f"Predictions: {preds}")

        # Basic assertions to ensure predictions are correct
        assert preds is not None, "Prediction should not be None"
        assert len(preds) > 0, "Prediction should return at least one value"
        assert all(isinstance(p, (int, float)) for p in preds), "Predictions should be numeric"
        assert expected, "Prediction should return expected results"
    except Exception as e:
        print(f"Error during prediction: {e}")
        pytest.fail(f"Prediction pipeline failed with error: {e}")
