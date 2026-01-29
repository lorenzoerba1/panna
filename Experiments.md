# Expertiments Register

## Do we really need 1000 repetitions?
- **Date**: 2026-01-28
- **Git Hash**: `66bae24a855027aa0159ce7e08d2979b31f1ba13`
- **Command**: `sbatch jobfile.slurm`
- **Logs Prefix**: `monaco/logs/amoxicillina`

### Motivation
See what happends if we reduce the number of repetitions from 1000 to 500
on the current version of the code

### Data
```csv
dataset,exact,approx
census,,
ht,,
pamap2,,
chem,,
fashion,221.79,7.66
nytimes,4293.34,312.10
glove,,
sift,,
gist,,
simplewiki,,
landmark,,
imagenet,,
```

### Result
Yes we do, the rehashes are too costly [still running]



## Is the main working?
- **Date**: 2026-01-28
- **Git Hash**: `66bae24a855027aa0159ce7e08d2979b31f1ba13`
- **Command**: `sbatch jobfile.slurm`
- **Logs Prefix**: `monaco/logs/balenot`

### Motivation
Check the point of the current version of the code

### Data
```csv
dataset,exact,approx
census,1137.41,218.95
ht,5704.30,2789.55
pamap2,,
chem,,
fashion,31.59,9.21
nytimes,2794.01,164.72
glove,FAILED,
sift,1861.01,245.68
gist,FAILED,
simplewiki,13615.06,996.67
landmark,4368.69,153.68
imagenet,,
```

### Result
Fails on gist, glove

## Bloom Filter to avoid duplicate comparisons
- **Date**: 2026-01-27
- **Git Hash**: `4b5da83e6bc53151d43c106efa62065179bbae7c`
- **Command**: `sbatch jobfile.slurm`
- **Logs Prefix**: `monaco/logs/baleno`

### Motivation
Can we use a Bloom filter to avoid the comparison of pairs we already seen in other rehashes?
### Data
```csv
dataset,exact,approx
census,388.86,86.22
ht,,
pamap2,,
chem,,
fashion,28.69,4.16
nytimes,,
glove,,
sift,,
gist,,
simplewiki,,
landmark,7370.41,159.12
imagenet,,
```

### Result
I guess not!


## [Title of Experiment]
- **Date**: YYYY-MM-DD
- **Git Hash**: `[hash]`
- **Command**: `sbatch jobfile.slurm`
- **Logs Prefix**: `[name]`

### Motivation
[Description of why this experiment is being conducted]

### Data
```csv
dataset,exact,approx
census,,
ht,,
pamap2,,
chem,,
fashion,,
nytimes,,
glove,,
sift,,
gist,,
simplewiki,,
landmark,,
imagenet,,
```

### Result
[Sintesi dei risultati e conclusioni]
