import csv
import scipy.stats as st
import pandas as pd
import numpy as np
import mlflow

# Calcualtes T-test on two run_ids with a stats_df indexed by those run_ids
def winner_t_test(df, a, b, equal_var, alternative):
        mean1 = np.asarray(df.loc[a]["mean"], dtype="float64")
        mean2 = np.asarray(df.loc[b]["mean"], dtype="float64")
        nobs1 = 12811
        nobs2 = 12811

        std1 = np.asanyarray(np.sqrt(df.loc[a]["var"]), dtype="float64")
        std2 = np.asanyarray(np.sqrt(df.loc[b]["var"]), dtype="float64")
        #print(winner1, winner2)

        result = st.ttest_ind_from_stats(mean1, std1, nobs1, 
                                mean2, std2, nobs2, 
                                equal_var=equal_var, alternative=alternative)
        return result

if __name__ == "__main__":
    import sys

    mlflow.set_tracking_uri("/data/projects/learnable-stitching/mlruns")
    mlflow.set_experiment("learnable-stitching-v0.4")

    if len(sys.argv) == 2:
        data_file = str(sys.argv[1])
        pair = None
    elif len(sys.argv) > 3:
        data_file = str(sys.argv[1])
        pair = []
        pair.append(str(sys.argv[2]))
    else:
        print("No input file provided")
        exit(0)

    df = pd.read_csv(data_file)
    df_stitch = df[df["phase"] == "stitching"]
    df_finetune = df[df["phase"] == "downstream"].set_index("run_id")

    #print(df_stitch)
    to_file = False
    if pair is not None:
         run_ids = pair
    else: 
         run_ids = list(df_stitch["run_id"])
         to_file = True
         
    if to_file:
        with open("analysis/p-values.csv", "w") as file:
            file.write("key,stitch_pvalue,tune_pvalue")

    df_stitch = df_stitch.set_index("run_id")
    for winner1, winner2 in zip(*[iter(run_ids)]*2):
        mean1 = df_stitch.loc[winner1]["mean"]
        mean2 = df_stitch.loc[winner2]["mean"]
        nobs1 = 12811
        nobs2 = 12811

        std1 = np.sqrt(df_stitch.loc[winner1]["var"] / (nobs1 - 1))
        std2 = np.sqrt(df_stitch.loc[winner2]["var"] / (nobs2 - 1)) 
        #print(winner1, winner2)

        #want to show that the model that causes the rank order is greater when stiching and lesser when fine_tuning, according to loss
        stich_t_stat, stich_pvalue = winner_t_test(df_stitch, winner2, winner1, equal_var=False, alternative="greater")
        tune_t_stat, tune_pvalue= winner_t_test(df_finetune, winner2, winner1, equal_var=False, alternative="less")
        
        model_a = df_stitch.loc[winner2]["model_name"]
        model_b = df_stitch.loc[winner1]["model_name"]
        client = mlflow.MlflowClient()
        run = client.get_run(winner1)
        metric = df_stitch.loc[winner1]["metric_name"]
        downstream_model = run.data.params["donorB_model"] + "-" + run.data.params["donorB_layer"] + "-" + run.data.params["donorB_dataset"]
        

        if to_file:
            #write out the p-values in latex document format
            with open("analysis/p-values.csv", "a") as file:
                arch = ''.join(filter(lambda x: x.isdigit(), downstream_model.split("-")[0]))
                layer = ''.join(filter(lambda x: x.isdigit(), downstream_model.split("-")[1]))
                layer = "0" + layer if len(layer) == 1 else layer
                key = "r" + str(arch) + "_" + str(layer)

                sci_st_p = f"{stich_pvalue:.3e}".split("e")
                sci_tu_p = f"{tune_pvalue:.3e}".split("e")
                
                if sci_st_p[1][1] == "0":
                    sci_st_p[1] = sci_st_p[1][:1] + "" + sci_st_p[1][2:]
                if sci_tu_p[1][1] == "0":
                    sci_tu_p[1] = sci_tu_p[1][:1] + "" + sci_tu_p[1][2:]

                sci_st_p = "$ " + sci_st_p[0] + " \\times 10^" + "{" + sci_st_p[1] + "} $"

                sci_tu_p = "$ " + sci_tu_p[0] + " \\times 10^" + "{" + sci_tu_p[1] + "} $"
                file.write(f"\n{key},{sci_st_p},{sci_tu_p}")

        if (stich_pvalue < 0.00125) and (tune_pvalue < 0.00125):
            print(f"\nIndependent T-Test Results r.o.v. in {metric} Between {model_a} and {model_b} on downstream{downstream_model}")
            
            print(f"{model_a} {metric} > {model_b} {metric}")
            print(f"t-stat: {stich_t_stat}  p-value: {stich_pvalue}")

            print(f"{model_a} {metric} < {model_b} {metric}")
            print(f"t-stat: {tune_t_stat}  p-value: {tune_pvalue}")
            
        
         