# TODOs

* Implement something more elaborate (and less resource wasting) than simple grid searches.
* Allow *logical not* operator for `conditions`. For instance, if value of `arg1` is Not `-1`, then ...
* Automatically add quotes to strings in `grid`.
* Incorporate a logger into the hpsearch.
* Allow to delete the output directory of runs right away inside the script `hpsearch` if a performance criteria is not met.
* The postprocessing should allow grouping experiments based on their random seeds (i.e., all runs with the same config except for output folder and random seed) and provide mean and std for performance measures in this case.
* Allow sending a signal to the hpsearch that will stop all scheduled jobs submitted and kill the hpsearch (helpful in case a mistake was made in configuring the hpsearch).
* Add a script that gathers random seeds of a given (or the best) run. Therefore, you may utilize the option `--grid_config` and based on the simulation's config generate a new `grid`.
