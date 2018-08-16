import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Plot generator", description="Generates a plot from the suplied training data logs")
parser.add_argument("logs", nargs="+", metavar="FILE", help="Input log files.")
parser.add_argument("-o", "--output", help="Output file for the plot.")
parser.add_argument("-l", "--plot-loss", action="store_true", help="If set loss, will be plotted, otherwise accuracy will be plotted.")
parser.add_argument("-t", "--title", help="Plot title.")
parser.add_argument("-y", "--ylabel", help="Y axis label")
args = parser.parse_args()

metric = "loss" if args.plot_loss else "acc"
if args.ylabel == None:
    args.ylabel = metric.capitalize()
if args.title == None:
    args.title = "%s by training epoch." % metric.capitalize()

data = dict()
for file in args.logs:
    tag = file[:file.rindex(".")]
    f = open(file,"r")
    values = [tuple(l[:-1].split(": ")) for l in f.readlines()]
    data[tag] = {m: map(float,l.split(" ")) for (m,l) in values}
    f.close()

N = len(data.values()[0]["acc"])

plt.style.use("ggplot")
plt.figure()
for tag, metrics in data.items():
    p = plt.plot(np.arange(0, N), metrics[metric], label="%s %s"%(tag,metric))
    color = p[0].get_color();
    plt.plot(np.arange(0, N), metrics["val_"+metric], label="%s %s"%(tag,"val_"+metric), color=color, dashes=[2,2])
plt.title(args.title)
plt.xlabel("Epoch #")
plt.ylabel(args.ylabel)
loc = "lower left" if not args.plot_loss else "upper right"
plt.legend(loc=loc)
plt.savefig(args.output)
