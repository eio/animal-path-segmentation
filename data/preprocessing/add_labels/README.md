# Data Pre-processing

## Segmentation Labels

Headers:

	["", "Identifier", "Species", "Individual", "Status", "Date"]

- `"Date"` = the transition date from one status to another
- Summer and Winter seasons are implicit (i.e. not explicitly labeled)
- If the timeseries starts/ends mid-season (basically always does) you only have the end/start of a season

Segmentation labels can have one of the following `Status` values:

	# Unique status strings: `df.Status.unique()`
	POSSIBLE_STATES = [
	    "Start Fall",
	    "End Fall",
	    "Start Spring",
	    "End Spring",
	    "Presumed Start Fall",
	    "Presumed End Fall",
	    "Presumed Start Spring",
	    "Presumed End Spring",
	    # Stopovers are called out.
	    # For now, maybe gloss over these
	    #   (or make a secondary behavioral state that captures those.)
	    "Start Stopover",
	    "End Stopover",
	    # This status value is not present in any of the usable data
	    # after the un-usable data has been purged
	    "No migration",
	    # Records with `Not enough data` are purged on load
	    "Not enough data",
	]

Unique species with labels:

	['Anthropoides paradiseus', 'Anthropoides virgo', 'Grus grus', 'Grus nigricollis', 'Grus vipio']
    ...and also ['Balearica pavonina'] but there's "Not enough data" for any of those records.

To inspect records for an individual:

	df_labels.loc[df_labels.Individual == 4125423]


## Events

Important headers:
	
	lat, lon, event_id, individual_id, timestamp, tag_id, taxon_canonical_name, species

Unique species with events: `list(df.species.unique())`
	
	['Anthropoides paradiseus', 'Anthropoides virgo', 'Grus grus', 'Grus nigricollis', 'Grus vipio']