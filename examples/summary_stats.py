import numpy as np
import itertools
def make_sum_stats(fields):

    def gen_summary_stats(simulation):
        sum_stats = {}

        for field in fields:
            
            this_sim = np.asarray(simulation[field])
            
            sum_stats[field+"_xmax_diff"] = np.argmax(np.diff(this_sim))
            sum_stats[field+"_ymax_diff"] = np.max(np.diff(this_sim))
            sum_stats[field+"_xmax"] = np.argmax(this_sim)
            sum_stats[field+"_ymax"] = np.max(this_sim)
            sum_stats[field+"_avg_growth_rate"] = np.average(np.diff(this_sim[:np.argmax(this_sim)]))
            sum_stats[field+"_xstart"] = np.argmax(this_sim != 0)

            sum_stats[field+"_val_end"] = this_sim[-1]

        for field0, field1 in itertools.product(fields, fields):
            if field0 == field1:
                continue
            corr = np.correlate(
                simulation[field0], simulation[field1], mode="same") 
            corr = corr /(np.std(simulation[field0])*np.std(simulation[field1]))
            sum_stats["x_{}_{}_ymax".format(field0, field1)] = np.max(corr)
            sum_stats["x_{}_{}_xmax".format(field0, field1)] = np.argmax(corr)
        return sum_stats
    return gen_summary_stats