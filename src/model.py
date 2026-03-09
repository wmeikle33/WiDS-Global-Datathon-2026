def xgb_pipeline:
		xgb_params = {
	    'n_estimators': 200,
	    'max_depth': 3,
	    'learning_rate': 0.05,
	    'random_state': 42
	}

	base_xgb = xgb.XGBClassifier(**xgb_params)

	calibrated_xgb = CalibratedClassifierCV(base_xgb, cv=5)

	Model = MultiOutputClassifier(calibrated_xgb)