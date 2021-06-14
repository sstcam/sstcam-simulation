
## 1_define_cameras.py

Creates the camera definition files (e.g. pulse shape)

```
p 1_define_cameras.py -o cameras
```

## 2s_simulate_events.py

Job submission script for simulating the events

```
p 2s_simulate_events.py -i /lfs/l2/chec/userspace/jasonjw/Data/sim_telarray/d200616_prod4/*/*_pe.h5 -c /lfs/l2/chec/Software/sstcam-simulation/optimisation_studies/pulse_shape2/cameras/*.pkl
```

## 3s_summarise_runs.py

Job submission script for summarising each run in terms of their performance curves (trigger and charge)

```
p 3s_summarise_runs.py -i cameras/*_events.h5
```

## 4_extract_performance.py

Combine together the summaries and extract a couple of key points from the performance curves

```
p 4_extract_performance.py -i cameras/*_performance.h5
```

## 5_plot_performance.py

Create performance plots
