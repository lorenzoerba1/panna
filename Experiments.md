# Expertiments Register

## Changes on old code
- **Date**: 2026-02-12
- **Git Hash**: `fd5b2273ad7f51e50a5432a1154c757561650753`
- **Command**: `sbatch jobfile.slurm`
- **Logs Prefix**: `monaco/logs/frollino`

### Motivation
Implemented the lookup of the last hash prefix to avoid comparisons in the rehashes, plus some small random tweaks

### Data
```csv
dataset,exact,approx
census,74.48,11.63
ht,638.34,555.61
pamap2,,
chem,,
fashion,6.14,4.54
nytimes,22.19,14.33
glove,321.78,97.14
sift,93.32,35.29
gist,,
simplewiki,83.02,26.93
landmark,260.61,53.42
imagenet,238.27,70.62
```

### Result
Great results, running now with an expect on (is_connected) to see if it actually works


## Some modifications to the code
- **Date**: 2026-02-12
- **Git Hash**: `d79309db26925ed8e7aa6eadf057e8bea54ed02c`
- **Command**: `sbatch jobfile.slurm`
- **Logs Prefix**: `monaco/logs/spring`

### Motivation
Implemented the lookup of the last hash prefix to avoid comparisons in the rehashes, plus some small random tweaks

### Data
```csv
dataset,exact,approx
census,344.30,119.68
ht,589.74,587.32
pamap2,3103.90,1723.80
chem,,
fashion,41.23,6.17
nytimes,1809.88,106.79
glove,11481.90,7344.73
sift,1443.00,1001.30
gist,,
simplewiki,16654.59,1001.30
landmark,3061.66,464.62
imagenet,1443.00,162.48
```

### Result
It's finally working on all datasets, but it's kinda slow

## So there's this very old version
- **Date**: 2026-01-30
- **Git Hash**: `66f47413d136228bbbc1d6d173c864f8c1e32d91`
- **Command**: `sbatch jobfile.slurm`
- **Logs Prefix**: `monaco/logs/cefixima`

### Motivation
Bisect to see where the performance degradation started

### Data
```csv
dataset,exact,approx
census,25.14,26.22
ht,NOT CONNECTED,
pamap2,371.82,365.28
chem,,
fashion,15.26,8.25
nytimes,1758.25,118.59
glove,,
sift,2323.33,142.49
gist,,
simplewiki,,
landmark,,
imagenet,,
```

### Result
[still running]


## So there's this old version
- **Date**: 2026-01-29
- **Git Hash**: `7bbcd6ee171154af87b1bcf13a9659183892d750`
- **Command**: `sbatch jobfile.slurm`
- **Logs Prefix**: `monaco/logs/azitromicina`

### Motivation
Bisect to see where the performance degradation started

### Data
```csv
dataset,exact,approx
census,,
ht,,
pamap2,,
chem,,
fashion,32.93,5.78
nytimes,1709.60,370.79
glove,TIMEOUT,
sift,5126.96,165.68
gist,,
simplewiki,15747.65,1221.39
landmark,,
imagenet,,
```

### Result
Bad version again


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
