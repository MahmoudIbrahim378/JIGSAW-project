1. Executive Summary
This project implements a jigsaw-style puzzle reconstruction system using only classical image processing methods (no AI/ML models). The pipeline is split into two milestones:
• Milestone 1 extracts puzzle pieces (or tiles), produces masks/crops, and exports metadata.
• Milestone 2 compares piece borders to propose likely neighbors and produces a repaired/assembled image with visual match explanations.
2. Problem Statement and Constraints
Goal: Given an input stream of puzzle images, extract pieces and then assemble the image by identifying likely neighboring pieces.
Constraints:
• Use only classical computer vision and image processing (OpenCV/Numpy). No deep learning or ML.
• Provide visualizations of candidate matches.
• Provide clear documentation of the pipeline and design decisions.
3. Inputs and Assumptions
We primarily tested on tray/grid-style puzzles (e.g., 2x2 and 3x3) where the puzzle is presented as a single image containing tiles separated by black gaps. The solution is still general in structure (piece extraction + matching), but the matching/assembly method is optimized for tile-like pieces.
Key practical assumption for robust extraction in our dataset: black separators/padding may be present and should be removed before edge matching to avoid degenerate matches.
4. Milestone 1: Piece Extraction and Metadata Export
 
4.1 Pipeline Overview
Milestone 1 takes puzzle images and outputs, for each detected piece:
• a binary mask
• a cropped RGB image
• the piece contour and smoothed contour
• curvature-based edge segmentation (contour split into edge segments)
• metadata saved to JSON/CSV for Milestone 2
4.2 Tray/Grid Split Mode (Fix for 2x2 / 3x3 Tray Images)
Problem observed: when the input is a single tray image (multiple tiles on a black background), contour detection often returns one large contour (the entire tray), producing only one 'piece'.
Solution: a preprocessing step splits the tray into ROWS x COLS tiles, optionally trims separators, and adds a small black border to each tile so that each tile becomes an independent input image for the original Milestone-1 pipeline.
4.3 Image Enhancement and Segmentation
The extraction pipeline uses the following processing order:
1) CLAHE contrast enhancement (LAB space)
2) Denoising (median blur)
3) Gamma correction
4) Convert to grayscale
5) Otsu thresholding (binary inverse)
6) Morphological open/close to clean noise and fill gaps
7) Contour detection and area filtering to keep valid pieces
4.4 Edge Representation (Contour Smoothing + Curvature Splitting)
For each piece contour we apply a moving-average smoothing step to reduce pixel noise. We then estimate discrete curvature along the contour and split the contour into segments using curvature peaks. These segments can be used as candidate 'edges' for more advanced jigsaw matching; in our tray/tile matching we rely primarily on border strips, but we keep this representation to satisfy the requirement of edge representation output.
4.5 Milestone-1 Outputs (Folder Structure)
Milestone 1 writes outputs under puzzle_output/ with the following structure:
• puzzle_output/pieces_metadata.json  (all pieces)
• puzzle_output/pieces_summary.csv    (summary table)
• puzzle_output/<image_name>/
    - masks/   (mask per piece)
    - crops/   (crop per piece)
    - vis/     (contours and edge visualizations)
    - debug/   (pipeline debug montage)
5. Milestone 2: Neighbor Matching and Assembly
 
 
(after changing grid to 3 and declare this in milestone2)
5.1 Why We Changed the Matching Strategy
Initial attempts that compared unordered edge pixels produced degenerate results on tray puzzles because many border pixels were nearly identical (black separators), leading to many zero or near-zero match scores.
We therefore introduced two key improvements:
• Fix A: Trim black borders/separators in addition to mask trimming.
• Fix B: Use ordered edge profiles (1D RGB signals along each border) instead of unordered pixel sets.
5.2 Fix A: Robust Border Trimming
Each tile is first trimmed by its mask bounding box (removes background/padding from Milestone 1 crops). Then, a second trimming step removes remaining black separators by cropping to pixels above a small grayscale threshold. This prevents every tile from sharing nearly identical black edge strips.
5.3 Fix B: Edge Profiles and Distance Metric
For each tile and each side (top/right/bottom/left), we compute an ordered edge profile:
• Take a strip of thickness k pixels from that side.
• Average across strip thickness to get a 1D RGB profile.
To compare two candidate neighbors, we compute mean absolute difference (L1) between the two 1D profiles after resizing them to the same length. Lower scores indicate a better match.
5.4 Global Assembly (Better Than Greedy and Cheaper Than Brute Force)
To assemble an RxC grid, we define the total grid cost as the sum of:
• right_cost(A,B) for each horizontal adjacency A→B
• down_cost(A,B)  for each vertical adjacency A→B

Instead of greedy placement (which can get stuck) or brute force (factorial explosion), we use:
• Beam Search: keeps the best B partial placements while filling the grid left-to-right, top-to-bottom.
• Simulated Annealing refinement: performs random tile swaps and accepts worse moves with decreasing probability to escape local minima and reduce total grid cost.
5.5 Visualization of Matches (Required Output)
After assembly, we visualize candidate matches directly on the assembled image:
• Draw lines between neighboring tile centers.
• Annotate each line with the adjacency score.
This provides an interpretable explanation of why the system selected those neighbors.
5.6 Milestone-2 Outputs
Milestone 2 saves:
• milestone2_output/assembled.png  (final assembled image)
• milestone2_output/grid_order.txt (tile IDs in final arrangement)
6. Key Parameters and Suggested Settings
Parameter	Used In	Typical Value	Effect / Notes
GRID_ROWS, GRID_COLS	M1 tray split	2/2 or 3/3	Set based on puzzle grid size.
SPLIT_MARGIN_RATIO	M1 tray split	0.02	Trims separators between tiles.
PAD_BORDER	M1 tray split	10	Adds black padding; helps segmentation, must be removed in M2.
MIN_AREA_RATIO	M1 contour filter	0.002	Filters tiny contours/noise.
k (strip thickness)	M2 edge profiles	8	Thicker strips reduce noise but may blur details.
TOPK	M2 pruning	12 (20 for 3*3)	Limits candidates per adjacency for speed.
BEAM_WIDTH	M2 beam search	250- (800 for 3*3)	Larger improves quality; slower.
CAND_LIMIT	M2 beam search	25 (increase t0 40-45 for 3*3)	Candidates per position after pruning.
Anneal iters	M2 annealing	15000	More iterations can improve results.
7. How to Run (Google Colab)
Milestone 1:
1) Upload images into input_images/.
2) Set TRAY_SPLIT_MODE and GRID_ROWS/GRID_COLS if using a single tray image.
3) Run milestone1_pipeline.py. Verify puzzle_output/pieces_metadata.json is created.
Milestone 2:
1) Set GRID_R and GRID_C to match the puzzle size.
2) Run milestone2_pipeline.py.
3) Check milestone2_output/assembled.png and the match visualization figure.
8. Limitations and Future Improvements
Limitations:
• The current scoring is optimized for tray/tile puzzles; true jigsaw pieces with tabs/holes would benefit from shape-based compatibility checks (e.g., curvature sign, edge-length normalization, and orientation alignment).
• Beam search + annealing may require parameter tuning for large grids.
Note: parameters changeable between 3*3 and 2*2

9. References
1) OpenCV Documentation: Image Thresholding, Morphological Operations, Contours.
2) Tut4,,5,6
3) youtube : https://www.youtube.com/watch?v=6rhBRtaxGhk

