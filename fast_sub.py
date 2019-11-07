import pandas as pd


s = "../input/...../submission.csv"
predict = pd.read_csv(s)

try:
    test_df = pd.read_csv("../input/severstal-steel-defect-detection/sample_submission.csv")
except:
    test_df = pd.read_csv("../input/test.csv")

test_df = pd.DataFrame(test_df['ImageId_ClassId'])

sub = pd.merge(test_df, predict, on='ImageId_ClassId', how='left')
sub.to_csv("submission.csv", index=False)
