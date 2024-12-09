import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

import statsmodels.api as sm
from statsmodels.formula.api import ols

import pingouin as pg

from openlockagents.common.log_io import filter_dataframe


def plot_number_of_attempts(data_pd, plot_filename, x_col="trial_num", hue="trial_scenario_name", ylim=40):
    ax = sns.barplot(x=x_col, y="trial_attempt_count", hue=hue, data=data_pd, errorbar='se', errwidth=1.5, capsize=0.05)
    ax.set(ylim=(0, ylim))
    plt.savefig(plot_filename)
    plt.close()


def run_ttests(group1_data, group2_data):

    group1_max_trial = group1_data["trial_num"].max()
    group2_max_trial = group2_data["trial_num"].max()
    assert_str = "group1 and group2 should have same number of trials"
    assert group1_max_trial == group2_max_trial, assert_str
    max_trial = group1_max_trial

    significances = dict()
    for t_idx in range(max_trial+1):
        trial_group1_data = filter_dataframe(group1_data, col="trial_num", values=[t_idx])
        trial_group2_data = filter_dataframe(group2_data, col="trial_num", values=[t_idx])
        trial_group1_attempt_count = trial_group1_data.trial_attempt_count
        trial_group2_attempt_count = trial_group2_data.trial_attempt_count
        ttest = stats.ttest_ind(trial_group1_attempt_count, trial_group2_attempt_count)
        significances["trial" + str(t_idx)] = ttest

    return significances


def deprecated_run_anova_training(training_df):
    df = training_df.copy()
    df["trial_num"] = df["trial_num"].astype(str)
    df["trial_num"] = pd.Categorical(df["trial_num"])
    df["trial_scenario_name"] = pd.Categorical(df["trial_scenario_name"])
    model = ols('trial_attempt_count ~ C(trial_num) + C(trial_scenario_name)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table


def run_anova_testing(testing_df):
    df = testing_df.copy()
    df["train_scenario_name"] = pd.Categorical(df["train_scenario_name"])
    df["test_scenario_name"] = pd.Categorical(df["test_scenario_name"])
    model = ols('trial_attempt_count ~ C(train_scenario_name) + C(test_scenario_name)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table


def run_anova_training(training_df):
    df = training_df.copy()
    df["trial_num"] = df["trial_num"].astype(str)
    df["trial_num"] = pd.Categorical(df["trial_num"])
    df["trial_scenario_name"] = pd.Categorical(df["trial_scenario_name"])
    anova_table = pg.mixed_anova(dv='trial_attempt_count', within='trial_num', between='trial_scenario_name', subject='subject_id', data=df)
    return anova_table


def run_anova_training_by_subject(training_df):
    df = training_df.copy()
    df["trial_num"] = df["trial_num"].astype(str)
    df["trial_num"] = pd.Categorical(df["trial_num"])
    df["trial_scenario_name"] = pd.Categorical(df["trial_scenario_name"])
    anova_table = pg.rm_anova(dv='trial_attempt_count', within='trial_num', subject='subject_id', data=df)
    return anova_table
