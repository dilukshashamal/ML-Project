import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from src.pipeline.predict_pipeline import PredicPipeline, CustomData
from src.pipeline.train_pipeline import TrainPipeline

@pytest.mark.training
def test_train_pipeline():
    try:
        pipeline = TrainPipeline()
        pipeline.start_training_pipeline()
        assert True, "Train pipeline ran successfully"
    except Exception as e:
        pytest.fail(f"Train pipeline failed with error: {e}")

@pytest.mark.prediction
def test_predict_pipeline():
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

        predict_pipeline = PredicPipeline()
        preds = predict_pipeline.predict(data_frame)

        assert preds is not None, "Prediction should not be None"
        assert len(preds) > 0, "Prediction should return at least one value"
    except Exception as e:
        pytest.fail(f"Prediction pipeline failed with error: {e}")
