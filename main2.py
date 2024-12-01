import pandas as pd
import pycaret
from pycaret.regression import *
import matplotlib.pyplot as plt
from pycaret.datasets import get_data  # Added this import
import matplotlib
matplotlib.use('Agg')  # Or 'Qt5Agg' or another interactive backend
import matplotlib.pyplot as plt

class ParkinsonsUPDRSComprehensiveAnalysis:
    def __init__(self, dataset_name='parkinsons_updrs'):
        # Load the dataset
        self.data = get_data(dataset_name)
        
        # Remove motor_UPDRS as specified
        self.data = self.data.drop('motor_UPDRS', axis=1)
        print("\nDrop motor_UPDRS...")
        self.data.head()
        
        # Experiment tracking
        self.experiment = None
        self.best_model = None
        self.catBoost = None

    
    def run_comprehensive_analysis(self):

        def catBoostModel(self):
        # Modify plot handling

            self.catBoost = create_model('catboost')
        
        # Save plots instead of displaying
            try:
                # Residuals plot
                plt.figure(figsize=(10, 6))
                plot_model(self.catBoost, plot='residuals', save=True)
                plt.close()  # Explicitly close the figure
                
                # Error plot
                plot_model(self.catBoost, plot='error', save=True)
                plt.close()
                
                # Feature importance plot
                plot_model(self.catBoost, plot='feature', save=True)
                plt.close()
            except Exception as e:
                print(f"Error in CatBoost plotting: {e}")
            
            # Rest of the existing code
                evaluate_model(self.catBoost)
                holdout_pred = predict_model(self.catBoost)
                print("\nPrediction dataframe")
                holdout_pred.head()
                save_model(self.catBoost, 'my_first_pipeline')
                loaded_best_pipeline = load_model('my_first_pipeline')
                return loaded_best_pipeline

        # Setup the experiment
        print("1. Setting up the experiment...")
        self.experiment = setup(self.data, 
                                target='total_UPDRS', 
                                normalize=True, 
                                normalize_method='minmax',
                                session_id=123)
        
        # Compare models
        print("\n2. Comparing models...")
        compare_tree_models = compare_models(
            include=['dt', 'rf', 'et', 'gbr', 'xgboost', 'lightgbm', 'catboost']
        )

        # Pull and print model comparison results
        compare_tree_models_results = pull()
        print(compare_tree_models_results)
        
        # Log the experiment
        print("\n3. Logging experiment...")
        save_experiment('my_experiment')
        
        #Create a cat boost to satisfy requirement
        print("\n3.5. Create a catboost model and display its performance by plotting important information")
        catBoostModel(self)

        # Create a model
        print("\n4. Creating a model...")
        self.best_model = compare_tree_models[0]
        
        # Tune the model
        print("\n5. Tuning the model...")
        tuned_model = tune_model(self.best_model, optimize='MAE')
        
        # Ensemble methods
        print("\n6. Ensemble modeling...")
        ensemble_bagging = ensemble_model(tuned_model, method='Bagging')
        ensemble_boosting = ensemble_model(tuned_model, method='Boosting')
        
        # Blend models
        print("\n7. Blending models...")
        blend_top_models = blend_models(compare_tree_models[:3])
        
        # Stack models
        print("\n8. Stacking models...")
        stack_top_models = stack_models(compare_tree_models[:3])
        
        # Model interpretation
        print("\n9. Model interpretation...")
        try:
            plt.figure(figsize=(10, 6))
            plot_model(tuned_model, plot='feature', save=True)
            plt.close()
        except Exception as e:
            print(f"Feature importance plot error: {e}")
        
        # Get leaderboard
        print("\n10. Retrieving leaderboard...")
        leaderboard = get_leaderboard()
        print(leaderboard)
        
        # AutoML
        print("\n11. Running AutoML...")
        automl_result = automl()
        
        # Create dashboard
        print("\n12. Creating dashboard...")
        dashboard(tuned_model, display_format='inline')
        
        # Create APP
        print("\n13. Creating APP...")
        create_app(tuned_model)
        
        # Create API
        print("\n14. Creating API...")
        create_api(tuned_model, api_name='parkinsons_updrs_api')
        
        # Create Docker
        print("\n15. Creating Docker...")
        create_docker('parkinsons_updrs_api')
        
        # Finalize model
        print("\n16. Finalizing model...")
        final_model = finalize_model(tuned_model)
        
        # Plot model (Addition to Cat Boost performance.)
        print("\n17. Plotting model...")
        try:
            plt.figure(figsize=(10, 6))
            plot_model(final_model, plot='residuals', save=True)
            plt.close()
            plot_model(final_model, plot='error', save=True)
            plt.close()
        except Exception as e:
            print(f"Model plots error: {e}")
            
        # Save model
        print("\n18. Saving model...")
        save_model(final_model, 'parkinsons_updrs_final_model')
        
        # Print data at the end
        print("\n19. Dataset Details:")
        print(self.data)
        
        return final_model


def main():
    analysis = ParkinsonsUPDRSComprehensiveAnalysis()
    final_model = analysis.run_comprehensive_analysis()

if __name__ == "__main__":
    main()