import pycaret
from pycaret.regression import *
from pycaret.regression import RegressionExperiment
from pycaret.datasets import get_data


print(pycaret.__version__)

from pycaret.datasets import get_data
data = get_data('parkinsons_updrs')

#print(data) #works

data = data.drop('motor_UPDRS', axis=1) #works
 
#print(data.columns)

s = setup(data, target = 'total_UPDRS')#, session_id = 123)

exp = RegressionExperiment()

type(exp)

exp.setup(data, target = 'total_UPDRS')#, session_id = 123)

catBoost = exp.compare_models()

catBoost = create_model('catboost')

evaluate_model(catBoost)

holdout_pred = predict_model(catBoost)

holdout_pred.head()

new_data = data.copy()
new_data.drop('total_UPDRS', axis=1, inplace=True)
new_data.head()

predictions = predict_model(catBoost, data = new_data)
predictions.head()

save_model(catBoost, 'my_first_pipeline')

loaded_best_pipeline = load_model('my_first_pipeline')
loaded_best_pipeline

#plot_model(catBoost, plot = 'residuals')

#plot_model(catBoost, plot = 'error')

#plot_model(catBoost, plot = 'feature')


s = setup(data, target = 'total_UPDRS')#, session_id = 123)

get_config()

get_config('X_train_transformed')

print("The current seed is: {}".format(get_config('seed')))

# now lets change it using set_config
set_config('seed', 786)
print("The new seed is: {}".format(get_config('seed')))

s = setup(data, target = 'total_UPDRS', #session_id = 123,
          normalize = True, normalize_method = 'minmax')

get_config('X_train_transformed')['age'].hist()

get_config('X_train')['age'].hist()

best = compare_models()

models()

compare_tree_models = compare_models(include=['dt', 'rf', 'et', 'gbr', 'xgboost', 'lightgbm', 'catboost'])

compare_tree_models

compare_tree_models_results = pull()
compare_tree_models_results

best_mae_models_top3 = compare_models(sort = 'MAE', n_select = 3)

best_mae_models_top3

models()

lr = create_model('lr')

lr_results = pull()
print(type(lr_results))
lr_results

lr = create_model('lr', fold=3)

create_model('lr', fit_intercept = False)

create_model('lr', return_train_score=True)

dt = create_model('dt')

tuned_dt = tune_model(dt)

dt

dt_grid = {'max_depth' : [None, 2, 4, 6, 8, 10, 12]}

tuned_dt = tune_model(dt, custom_grid = dt_grid, optimize = 'MAE')

tuned_dt, tuner = tune_model(dt, return_tuner=True)

tuned_dt

tuner

tuned_dt = tune_model(dt, search_library = 'optuna')

ensemble_model(dt, method = 'Bagging')

ensemble_model(dt, method = 'Boosting')


best_mae_models_top3

blend_models(best_mae_models_top3)

stack_models(best_mae_models_top3)

lightgbm = create_model('lightgbm')

#interpret_model(lightgbm, plot = 'summary')

lb = get_leaderboard()
lb

lb.sort_values(by='MAE', ascending=True)['Model'].iloc[0]

automl()


dashboard(dt, display_format ='inline')

create_app(best)

create_api(best, api_name = 'my_first_api')

create_docker('my_first_api')

final_best = finalize_model(best)

final_best

print(convert_model(dt, language = 'java'))

#deploy_model(best, model_name = 'my_first_platform_on_aws', platform = 'aws', authentication = {'bucket' : 'pycaret-test'})

#loaded_from_aws = load_model(model_name = 'my_first_platform_on_aws', platform = 'aws', authentication = {'bucket' : 'pycaret-test'})

#loaded_from_aws

save_model(best, 'my_first_model')

loaded_from_disk = load_model('my_first_model')
loaded_from_disk

save_experiment('my_experiment')

exp_from_disk = load_experiment('my_experiment', data=data)