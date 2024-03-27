SEED = 9582

model_params = {
    'reg_alpha': 0.0008774661176012108,
    'reg_lambda': 2.542812743920178,
    'colsample_bynode': 0.7839026197349153,
    'subsample': 0.8994226268096415, 
    # subsample=1,
    'eta': 0.04730766698056879, 
    'max_depth': 3, 
    'n_estimators': 500,
    'random_state': SEED,
    'eval_metric': 'rmse',
    'n_jobs': -1,
    'learning_rate':0.023,
}


class XGBoostModel:
    def __init__(self, data_X, data_Y, params):
        self.params = params
        self.X = data_X
        self.Y = data_Y
        self.models = []
        self.num_models = 5 # The reason why we want 


    def fit():
        for

