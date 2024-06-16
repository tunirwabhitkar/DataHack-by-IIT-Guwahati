# Predict on test data
test_preds = best_clf.predict_proba(X_test_preprocessed)

# Prepare the submission file
submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': [prob[1] for prob in test_preds[0]],
    'seasonal_vaccine': [prob[1] for prob in test_preds[1]]
})

submission.to_csv('D:/Hack-a-thon/submission_format.csv', index=False)
