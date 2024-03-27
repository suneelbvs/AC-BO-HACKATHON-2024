# HACK: helper functions
# might delete them later
def gen_feats(dataset):
    dataset["num_oxygen"] = dataset["Drug"].str.count("O")
    return dataset
