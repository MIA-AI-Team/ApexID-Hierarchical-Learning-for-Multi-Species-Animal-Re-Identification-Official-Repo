from apexid.data.encounters import infer_bristol_encounter, infer_elephant_encounter


def test_bristol_session_and_trial_groups():
    assert infer_bristol_encounter("afia", "afia-5-img-78.jpg") == (
        "bristol_gorillas:afia:session_5",
        True,
    )
    assert infer_bristol_encounter("touni", "touni-ggl-trial-img-95.jpg") == (
        "bristol_gorillas:touni:trial",
        True,
    )


def test_elephant_filename_preserves_date_precision():
    assert infer_elephant_encounter("ELPephants_1909", "1909_Jamila_13Oct2016.png") == (
        "elephants:ELPephants_1909:2016-10-13",
        True,
    )
    assert infer_elephant_encounter("ELPephants_35", "35_Malak_left_Jan2003.png") == (
        "elephants:ELPephants_35:2003-01",
        True,
    )
    assert infer_elephant_encounter("ELPephants_492", "492_Barrymore.png") == (
        "elephants:ELPephants_492:unknown_date",
        False,
    )
