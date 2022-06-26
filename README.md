# hots_classes
Classifying Heroes of the Storm characters by server stats using ML

Subject:
- HOTS (Heroes of the Storm) is a Blizzard franchise MOBA (https://heroesofthestorm.fandom.com)
- There are 90 playable characters referred to as heroes.
- Each hero has internal stats (HP, damage per second...etc) that dictate an optimal in-game role.
- The main roles are:
  -Front line: lead the charge in hero-hero fights
  -Damage dealer: deal heavy damage against heros
  -Support: provide support to allies
  -Side laner: deal damage to structures and periodically spawning minions

Goal:
Attempt to classify heroes into their roles by feeding server performance stats (average hero damage, average tanked damage...etc) into a machine learning model.

Data:
External stats (average player performance) from: hotslogs.com
Internal stats from: hotsnred.com

Method:
- This is not a controlled experiment: we don't have a way of calibrating our model. Instead we treat the results dynamically:
  - If a hero is clearly mis-classified, we try to improve the model
  - If a hero is oddly classified, we take a look at the stats: if they don't make sense, we improve the model, otherwise we accept the classification.
- First we will use k-means clustering.
- Then we will try to remove redundancies through dim-reg (dimensionality reduction).
- We will try PCA and autoencoders for dim-reg.

Files:
- XXX.html are raw webpages
- hots_log_parser.ipynb parses the html files to extract the data
- hots_stats.npz is the parsed server stats
- hots_stats_ext.npz includes internal stats
- norm_stats.npz is a normalized version
- hots_models.py contains autoencoder models
- hots_grid_i.py perform hyperparameter scans on the autoencoder models
- autoi_expj.pickle aregrid scan output
- exp_i.ipynb are the various experiments
