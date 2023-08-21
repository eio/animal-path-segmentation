import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Sample data
labels = [
    "Grus grus",
    "Anthropoides virgo",
    "Grus vipio",
    "Grus nigricollis",
    "Anthropoides paradiseus",
]
percentages = [
    60,
    137,
    30,
    12,
    6,
]
# Manually selected color scheme for colorblind friendliness
# https://davidmathlogic.com/colorblind/
colors = [
    "#785EF0",  # purple
    "#648FFF",  # bluish
    "#DC267F",  # pink #
    "#FE6100",  # orange #
    "#FFB000",  # yellow
]
# # "Tol" color palette
# other_colors = [
#     "#332288",
#     "#117733",
#     "#44AA99",
#     "#88CCEE",
#     "#DDCC77",
#     "#CC6677",
#     "#AA4499",
#     "#882255",
# ]

# # Sample data
# labels = ["Test", "Validation", "Train"]
# # Corresponding percentages
# percentages = [32, 21, 196]
# colors = [
#     "#FFC107",
#     "#D81B60",
#     "#1E88E5",
#     # "#004D40",
# ]
# colors = list(
#     reversed(
#         [
#             "#648FFF",  # bluish
#             "#785EF0",  # purple
#             # "#DC267F",  # pink #
#             # "#FE6100",  # orange #
#             "#FFB000",  # yellow
#         ]
#     )
# )


# Calculate the total sum of percentages
total_sum = sum(percentages)

# Create a pie chart
plt.figure(figsize=(6, 6))
# plt.title("Distribution of Categories")
wedges, texts, autotexts = plt.pie(
    percentages,
    labels=labels,
    colors=colors,
    autopct="%1.0f%%",
    startangle=140,  # Adjust the starting angle
    labeldistance=1.05,
    pctdistance=0.54,  # Adjust the distance of the labels from the center
    # startangle=140,
    # textprops=dict(color="w"),  # Set label text color to white
)

angle = 140
# Display the exact integer values and percentages in parentheses inside slices
for autotext in autotexts:
    # index = autotexts.index(autotext)
    # value = percentages[index]
    # autotext.set_text(f"{value} paths\n({(value / total_sum) * 100:.1f}%)")
    ###########################################
    index = autotexts.index(autotext)
    value = int(percentages[index])
    angle += (360 * value / total_sum) / 2  # Calculate middle angle of slice
    x = np.cos(np.radians(angle))
    y = np.sin(np.radians(angle))
    # Adjust the distance of the label from the center
    x_text = x * 0.82
    y_text = y * 0.84
    if value == 60:
        y_text = 0
    if value == 6 or value == 12:
        y_text = y * 0.9
    if value == 137:
        x_text = x * 0.7
        y_text = y * 0.25
    plt.text(
        x_text,
        y_text,
        # "{} paths\n({:.1f}%)".format(value, (value / total_sum) * 100),
        "{} paths".format(value),
        # color="w",
        ha="center",
        va="center",
        weight="bold",
    )
    angle += (360 * value / total_sum) / 2  # Move to the next middle angle


# Equal aspect ratio ensures that the pie is drawn as a circle.
plt.axis("equal")
# Adjust dpi and bbox_inches as needed
plt.savefig("pie_chart.png", dpi=150, bbox_inches="tight")
