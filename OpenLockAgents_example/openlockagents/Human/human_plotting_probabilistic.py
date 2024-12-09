import os

import pandas as pd
import seaborn as sns
import json
import argparse

import openlockagents.common.log_io as log_io
import openlockagents.Human.common as common
import openlockagents.common.data_analysis as data_analysis

## Cogsci 2018 data outliers by subject ID
#  outliers_bl = {'1581272937' '2437993833' '2537180140' '2887691837' '3000733472' '3622083161' '3636246095' '3651645526'};
#     outliers_xfer = {'2621512303' '2823189312' '2921021640' '3125513326' '3127842216' '3149972998' '3571817913'};
#     outliers_extras = {'2823606819' '1765495362' '2562371801' '2378760540' '2240960040' '3221124899' '3086567642' '2921063409' '3492951018' '3667837660' '1929833271' '3327419966' '2846933910' '2122378021' '2117379298'};

## Probabilistic OpenLock outliers by subject ID (as of 2020-09-02)
# # outliers_max_attempts = {"210916018587813701", "1381402771209201618", "1063439401621509469", "155909101124963913",
#                          "2107381161974581384"}
# # participant_id_corrections = {
# "965572421390697671": 122,
# "297314542793471609": 123,
# "786084844737664041": 124,
# "1768481991484224161": 155,
# "662060482887695737": 258,
# }

SCENARIO_TO_CONDITION = {
    "CE3-CE4": 1,
    "CE3-CC4": 2,
    "CC3-CE4": 3,
    "CC3-CC4": 4,
    "CE4": 5,
    "CC4": 6,
    1: "CE3-CE4",
    2: "CE3-CC4",
    3: "CC3-CE4",
    4: "CC3-CC4",
    5: "CE4",
    6: "CC4",
}


def main():
    parser = argparse.ArgumentParser("OpenLockBayesianLearner")
    parser.add_argument("--plotdir", type=str, help="directory to save the plots.")
    parser.add_argument("--datadir", type=str, help="directory to save the data.")
    args = parser.parse_args()

    sns.set_style("darkgrid")

    data_dir = args.datadir
    attempt_limit = 50

    subject_data_dir = data_dir
    plot_dir = args.plotdir

    outliers = set()
    outliers_max_attempts = {"210916018587813701", "1381402771209201618", "1063439401621509469", "155909101124963913", "2107381161974581384"}
    outliers = outliers.union(outliers_max_attempts)

    # corrections to fix human errors in the sign-in sheet
    participant_id_corrections = {
        "965572421390697671": 122,
        "297314542793471609": 123,
        "786084844737664041": 124,
        "1768481991484224161": 155,
        "662060482887695737": 258,
    }

    all_subject_data = log_io.load_subjects_from_dir(subject_data_dir, ext="pkl", participant_id_corrections=participant_id_corrections)

    all_subject_data_pd = log_io.build_dataframe(all_subject_data, outliers=outliers)

    group_counts = log_io.extract_group_counts(all_subject_data_pd)
    print("group counts:")
    print(group_counts)

    # sanity_checks(all_subject_data_pd, data_dir)

    baseline_pd = log_io.filter_dataframe(all_subject_data_pd, col="scenario_name", values=["CC4", "CE4"])
    transfer_pd = log_io.filter_dataframe(all_subject_data_pd, col="scenario_name", values=["CC3-CC4", "CC3-CE4", "CE3-CC4", "CE3-CE4"])

    data_analysis.plot_number_of_attempts(baseline_pd, plot_filename=plot_dir + "/baseline.png")

    # split the transfer cases into training and testing to plot
    transfer_training_pd = log_io.filter_dataframe(transfer_pd, col="trial_scenario_name", values=["CC3", "CE3"])
    transfer_testing_pd = log_io.filter_dataframe(transfer_pd, col="trial_scenario_name", values=["CC4", "CE4"])
    data_analysis.plot_number_of_attempts(transfer_training_pd, plot_filename=plot_dir + "/training.png", ylim=attempt_limit)
    data_analysis.plot_number_of_attempts(transfer_testing_pd, plot_filename=plot_dir + "/testing.png", x_col="test_scenario_name", hue="train_scenario_name", ylim=attempt_limit)

    # compute baseline t-tests
    cc4_training_data = log_io.filter_dataframe(baseline_pd, col="train_scenario_name", values=["CC4"])
    ce4_training_data = log_io.filter_dataframe(baseline_pd, col="train_scenario_name", values=["CE4"])
    baseline_significances = data_analysis.run_ttests(cc4_training_data, ce4_training_data)
    print("BASELINE SIGNIFICANCES:")
    pretty_print(baseline_significances)

    # compute transfer t-tests
    split_subject_data = log_io.split_df(all_subject_data_pd, column="scenario_name")
    cc3_cc4_transfer_data = split_subject_data["CC3-CC4"]
    cc3_ce4_transfer_data = split_subject_data["CC3-CE4"]
    ce3_cc4_transfer_data = split_subject_data["CE3-CC4"]
    ce3_ce4_transfer_data = split_subject_data["CE3-CE4"]
    cc4_transfer_significances = data_analysis.run_ttests(cc3_cc4_transfer_data, ce3_cc4_transfer_data)
    ce4_transfer_significances = data_analysis.run_ttests(ce3_ce4_transfer_data, cc3_ce4_transfer_data)
    print("TRANSFER SIGNIFICANCES:")
    print("CC4 TRANSFER SIGNIFICANCES:")
    pretty_print(cc4_transfer_significances)
    print("CE4 TRANSFER SIGNIFICANCES:")
    pretty_print(ce4_transfer_significances)

    all_subject_data_pd.to_excel(plot_dir + "/data_complete.xlsx")
    with pd.ExcelWriter(plot_dir + "/data_summary.xlsx") as writer:
        for key, split in split_subject_data.items():
            pivoted_subject_data = split.pivot(index="subject_id", columns="trial_num", values=["trial_attempt_count", "trial_scenario_name"])
            pivoted_subject_data.to_excel(writer, sheet_name=key)

    print("All done")


def check_participant_ids(subject_pd, data_dir):
    sign_in_sheet = data_dir + "/Probabilistic OpenLock Sign in sheet 1 - Sheet1.csv"
    sign_in_sheet_pd = pd.read_csv(sign_in_sheet)
    sign_in_sheet_pd = sign_in_sheet_pd[sign_in_sheet_pd["Condition"].notnull()]

    # rows where the sign-in sheet was valid
    completed_sign_in_sheet_pd = sign_in_sheet_pd[sign_in_sheet_pd["incomplete/outlier"] != "1"]

    participant_ids = sorted(subject_pd.participant_id.unique())
    print("participant IDs (total: {})".format(len(participant_ids)))
    print(participant_ids)

    # check for duplicate participant IDs
    participant_ids = subject_pd.participant_id.unique()
    for participant_id in participant_ids:
        participant_pd = subject_pd[subject_pd.participant_id == participant_id]
        subject_ids = participant_pd.subject_id.unique()
        assert_str = "Sanity check error: participant ID {} has multiple subject IDs associated with it: {}".format(participant_id, subject_ids)
        assert len(subject_ids) == 1, assert_str

    # check condition assignments
    for participant_id in participant_ids:
        participant_scenario = subject_pd[subject_pd.participant_id == participant_id].scenario_name.unique()
        assert_str = "Sanity check error: participant {} has multiple scenarios associated with it in data frame".format(participant_id)
        assert len(participant_scenario) == 1, assert_str

        participant_subject_id = subject_pd[subject_pd.participant_id == participant_id].subject_id.unique()
        assert_str = "Sanity check error: participant {} has multiple subject IDs {} associated with it in data frame".format(participant_id, participant_subject_id)
        assert len(participant_subject_id) == 1, assert_str

        participant_scenario = participant_scenario[0]
        participant_subject_id = participant_subject_id[0]

        participant_condition = SCENARIO_TO_CONDITION[participant_scenario]
        try:
            sign_in_sheet_condition = sign_in_sheet_pd[sign_in_sheet_pd.ID == participant_id].Condition.array[0]
        except IndexError:
            assert_str = "Error, index out of bounds for participant ID {}".format(participant_id)
            raise IndexError(assert_str)

        if sign_in_sheet_condition != participant_condition:
            print("\tmismatch scenario data: {}\t{}\t{}".format(participant_id, participant_scenario, SCENARIO_TO_CONDITION[participant_scenario]))
        assert_str = "Sanity check error: participant ID {}, subject ID {} in scenario {} has different condition ID in data vs. sign-in sheet: data: {} sign-in sheet: {}".format(
            participant_id, participant_subject_id, participant_scenario, participant_condition, sign_in_sheet_condition)
        assert sign_in_sheet_condition == participant_condition, assert_str

    # check that every valid sign-in sheet participant ID is in the data
    sign_in_sheet_participant_ids = list(completed_sign_in_sheet_pd.ID)
    missing_participant_ids = set(sign_in_sheet_participant_ids) - set(participant_ids)
    assert_str = "Sanity check error: participant IDs {} in sign-in sheet but not in data".format(missing_participant_ids)
    assert len(missing_participant_ids) == 0, assert_str

    # do the reverse check - verify every data participant ID is in the sign-in sheet
    missing_participant_ids = set(participant_ids) - set(sign_in_sheet_participant_ids)
    missing_participant_id_subject_ids = subject_pd[subject_pd["participant_id"].isin(missing_participant_ids)]["subject_id"]
    assert_str = "Sanity check error: participant IDs {} in the data but not in sign-in sheet".format(list(zip(missing_participant_ids, missing_participant_id_subject_ids)))
    assert len(missing_participant_ids) == 0, assert_str

    # check participant IDs missing from sign-in sheet were marked incomplete
    missing_participant_ids_from_data = set(sign_in_sheet_pd.ID) - set(participant_ids)
    for missing_participant_id in missing_participant_ids_from_data:
        incomplete_setting = sign_in_sheet_pd[sign_in_sheet_pd.ID == missing_participant_id]["incomplete/outlier"].array[0]
        assert_str = "Sanity check error: participant {} is not marked as an outlier/incomplete but is not in data".format(missing_participant_id)
        assert incomplete_setting == "1", assert_str

    # verify counts of condition IDs match
    # total_sign_in_group_counts is just to manually check against the sign-in sheet
    total_sign_in_group_counts = log_io.extract_group_counts(sign_in_sheet_pd, scenario_col_name="Condition", group_col_name="ID").to_dict()

    data_group_counts = log_io.extract_group_counts(subject_pd).to_dict()
    completed_sign_in_group_counts = log_io.extract_group_counts(completed_sign_in_sheet_pd, scenario_col_name="Condition", group_col_name="ID").to_dict()
    completed_sign_in_group_counts = dict([(SCENARIO_TO_CONDITION[cond], v) for cond, v in completed_sign_in_group_counts.items()])
    for key in data_group_counts.keys():
        assert_str = "Sanity check error: group {} has consistent count: data: {}, sign-in: {}".format(key, data_group_counts[key], completed_sign_in_group_counts[key])
        assert data_group_counts[key] == completed_sign_in_group_counts[key], assert_str


def sanity_checks(subject_pd, data_dir):
    '''
    sanity checks on the dataframe. Particularly, checks participant_id's for potential mismatches with the sign-in sheet
    :param subject_pd:
    :return:
    '''

    check_participant_ids(subject_pd, data_dir)

    print("All sanity checks passed")


def pretty_print(data):
    print(json.dumps(data, sort_keys=True, indent=4))



if __name__ == "__main__":
    main()