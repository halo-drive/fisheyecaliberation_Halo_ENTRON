import json
import numpy as np
import matplotlib.pyplot as plt

# Load the constraints
with open('./cam-2.json', 'r') as f:
    data = json.load(f)

# Image dimensions
img_width, img_height = data['image-size']
print(f"Image size: {img_width} x {img_height}")
print(f"Number of constraints: {len(data['constraints'])}")
print()

# Analyze each constraint's coverage
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, img_width)
ax.set_ylim(img_height, 0)  # Flip y-axis for image coordinates
ax.set_aspect('equal')
ax.set_title(f'Checkerboard Coverage - {len(data["constraints"])} Constraints')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')

# Draw grid to show image regions
for i in range(4):
    ax.axhline(y=i * img_height / 3, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=i * img_width / 3, color='gray', linestyle='--', alpha=0.3)

colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

all_x = []
all_y = []

for idx, constraint in enumerate(data['constraints']):
    frame_num = constraint['image-frame-number']
    points = np.array(constraint['image-points'])
    
    # Extract x, y coordinates
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    all_x.extend(x_coords)
    all_y.extend(y_coords)
    
    # Plot the checkerboard corners
    ax.scatter(x_coords, y_coords, c=colors[idx % len(colors)], 
               s=20, alpha=0.6, label=f'Frame {frame_num}')
    
    # Draw bounding box
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    print(f"Constraint {idx+1} (Frame {frame_num}):")
    print(f"  X range: {x_min:.1f} - {x_max:.1f} ({x_min/img_width*100:.1f}% - {x_max/img_width*100:.1f}%)")
    print(f"  Y range: {y_min:.1f} - {y_max:.1f} ({y_min/img_height*100:.1f}% - {y_max/img_height*100:.1f}%)")
    print(f"  Center: ({(x_min+x_max)/2:.1f}, {(y_min+y_max)/2:.1f})")
    print()

# Overall coverage
all_x = np.array(all_x)
all_y = np.array(all_y)

print("=" * 60)
print("OVERALL COVERAGE:")
print(f"  X range: {all_x.min():.1f} - {all_x.max():.1f} ({all_x.min()/img_width*100:.1f}% - {all_x.max()/img_width*100:.1f}%)")
print(f"  Y range: {all_y.min():.1f} - {all_y.max():.1f} ({all_y.min()/img_height*100:.1f}% - {all_y.max()/img_height*100:.1f}%)")
print()

# Check coverage of image regions
regions = {
    'Top-Left': (0, img_width/3, 0, img_height/3),
    'Top-Center': (img_width/3, 2*img_width/3, 0, img_height/3),
    'Top-Right': (2*img_width/3, img_width, 0, img_height/3),
    'Middle-Left': (0, img_width/3, img_height/3, 2*img_height/3),
    'Center': (img_width/3, 2*img_width/3, img_height/3, 2*img_height/3),
    'Middle-Right': (2*img_width/3, img_width, img_height/3, 2*img_height/3),
    'Bottom-Left': (0, img_width/3, 2*img_height/3, img_height),
    'Bottom-Center': (img_width/3, 2*img_width/3, 2*img_height/3, img_height),
    'Bottom-Right': (2*img_width/3, img_width, 2*img_height/3, img_height),
}

print("REGION COVERAGE:")
covered_regions = []
for region_name, (x_min, x_max, y_min, y_max) in regions.items():
    in_region = ((all_x >= x_min) & (all_x <= x_max) & 
                 (all_y >= y_min) & (all_y <= y_max)).any()
    status = "COVERED" if in_region else "MISSING"
    print(f"  {region_name:15s}: {status}")
    if in_region:
        covered_regions.append(region_name)

print()
print(f"Coverage: {len(covered_regions)}/9 regions covered ({len(covered_regions)/9*100:.1f}%)")
print()

# Assessment
print("=" * 60)
print("ASSESSMENT:")
print(f"  Constraints captured: {len(data['constraints'])}")
print(f"  Recommended minimum: 30-50 for fisheye")
print(f"  Status: {'INSUFFICIENT' if len(data['constraints']) < 20 else 'SUFFICIENT'}")
print()
print("MISSING CRITICAL AREAS:")
missing = [name for name in regions.keys() if name not in covered_regions]
for region in missing:
    print(f"  - {region}")

ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('./constraint_coverage.png', dpi=150, bbox_inches='tight')
print()
print("Coverage visualization saved to: constraint_coverage.png")
