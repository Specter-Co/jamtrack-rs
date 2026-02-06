# Define examples as a single array where each element is a space-separated string:
# "test_dir out_video start_frame end_frame sensor_id"
declare -a EXAMPLES=(
    # "0 np.inf 120 ./tracker_result_videos/test-1_tracks.mp4 ./data/specter-data-6-2/test-1/1748906723827.json ./data/specter-data-6-2/test-1/0_0.h265"
    # "0 np.inf 120 ./tracker_result_videos/test-2_tracks.mp4 ./data/specter-data-6-2/test-2/1748907694102.json ./data/specter-data-6-2/test-2/1748907554795_1748907565999.h265"
    # "0 np.inf 120 ./tracker_result_videos/test-3_tracks.mp4 ./data/specter-data-6-2/test-3/1748909216209.json ./data/specter-data-6-2/test-3/1748909066201_1748909075400.h265"
    
    # "0 np.inf 1004 ./tracker_result_videos/test-5-0x3ec_tracks.mp4 ./data/specter-data-6-12/1749754937516.json ./data/specter-data-6-12/0x3ec/1749754799638_1749754819234.h265"
    "0 np.inf 1005 ./tracker_result_videos/test-5-0x3ed_tracks.mp4 ./data/specter-data-6-12/1749754937516.json ./data/specter-data-6-12/0x3ed/1749754980367_1749754999166.h265"
    # "0 np.inf 1006 ./tracker_result_videos/test-5-0x3ee_tracks.mp4 ./data/specter-data-6-12/1749754937516.json ./data/specter-data-6-12/0x3ee/1749750812806_1749750832403.h265"
    # "0 np.inf 1007 ./tracker_result_videos/test-5-0x3ef_tracks.mp4 ./data/specter-data-6-12/1749754937516.json ./data/specter-data-6-12/0x3ef/1749748990066_1749749000866.h265"
    # "0 np.inf 1008 ./tracker_result_videos/test-5-0x3f0_tracks.mp4 ./data/specter-data-6-12/1749754937516.json ./data/specter-data-6-12/0x3f0/1749750611922_1749750622719.h265"

    # Does not work
    # "0 np.inf 1000 ./tracker_result_videos/test-4-1000_tracks.mp4 ./data/specter-data-6-6/1749270528552.json ./data/specter-data-6-6/_0x3e8/1749270528225_1749270546524.h265"
    # "0 np.inf 1001 ./tracker_result_videos/test-4-1001_tracks.mp4 ./data/specter-data-6-6/1749270528552.json ./data/specter-data-6-6/_0x3e9/1749270528509_1749270546309.h265"
    # "0 np.inf 1003 ./tracker_result_videos/test-4-1003_tracks.mp4 ./data/specter-data-6-6/1749270528552.json ./data/specter-data-6-6/_0x3eb/1749270528399_1749270546196.h265"
    # "0 np.inf 1004 ./tracker_result_videos/test-4-1004_tracks.mp4 ./data/specter-data-6-6/1749270528552.json ./data/specter-data-6-6/_0x3ec/1749270528321_1749270546120.h265"
    # "0 np.inf 1005 ./tracker_result_videos/test-4-1005_tracks.mp4 ./data/specter-data-6-6/1749270528552.json ./data/specter-data-6-6/_0x3ed/1749270528632_1749270545932.h265"

    #"0 500 120 ./tracker_result_videos/temp1_tracks.mp4 ./data/specter-data-6-2/test-2/1748907694102.json ./data/specter-data-6-2/test-2/1748907554795_1748907565999.h265"
    # "3000 3100 120 ./tracker_result_videos/temp1_tracks.mp4 ./data/specter-data-6-2/test-2/new_boxes_1088_1920.pt ./data/specter-data-6-2/test-2/1748907554795_1748907565999.h265"

    # Regened PCP
    # "0 np.inf 120 ./tracker_result_videos/test-1_tracks.mp4 ./data/specter-data-6-2/test-1/new_boxes_1088_1920.pt ./data/specter-data-6-2/test-1/0_0.h265"
    # "0 np.inf 120 ./tracker_result_videos/test-2_tracks.mp4 ./data/specter-data-6-2/test-2/new_boxes_1088_1920.pt ./data/specter-data-6-2/test-2/1748907554795_1748907565999.h265"
    # "0 np.inf 120 ./tracker_result_videos/test-3_tracks.mp4 ./data/specter-data-6-2/test-3/new_boxes_1088_1920.pt ./data/specter-data-6-2/test-3/1748909066201_1748909075400.h265"
    # "0 np.inf 1000 ./tracker_result_videos/test-4-1000_tracks.mp4 ./data/specter-data-6-6/_0x3e8/new_boxes_1088_1920.pt ./data/specter-data-6-6/_0x3e8/1749270528225_1749270546524.h265"
    # "0 np.inf 1001 ./tracker_result_videos/test-4-1001_tracks.mp4 ./data/specter-data-6-6/_0x3e9/new_boxes_1088_1920.pt ./data/specter-data-6-6/_0x3e9/1749270528509_1749270546309.h265"
    # "0 np.inf 1003 ./tracker_result_videos/test-4-1003_tracks.mp4 ./data/specter-data-6-6/_0x3eb/new_boxes_1088_1920.pt ./data/specter-data-6-6/_0x3eb/1749270528399_1749270546196.h265"
    # "0 np.inf 1004 ./tracker_result_videos/test-4-1004_tracks.mp4 ./data/specter-data-6-6/_0x3ec/new_boxes_1088_1920.pt ./data/specter-data-6-6/_0x3ec/1749270528321_1749270546120.h265"
    # "0 np.inf 1005 ./tracker_result_videos/test-4-1005_tracks.mp4 ./data/specter-data-6-6/_0x3ed/new_boxes_1088_1920.pt ./data/specter-data-6-6/_0x3ed/1749270528632_1749270545932.h265"
)

# Loop through all examples
for ((i=0; i<${#EXAMPLES[@]}; i++)); do
    # Split the example string into variables
    read -r START_FRAME END_FRAME SENSOR_ID OUT_VIDEO DET_PATH VIDEO_PATH <<< "${EXAMPLES[$i]}"
    
    EXAMPLE_NUM=$((i + 1))
    echo "Running example $EXAMPLE_NUM of ${#EXAMPLES[@]}"
    echo "----------------------------------------"

    # Run the Python script with the example parameters
    python3 python/resim_tracker.py \
        --video_path "$VIDEO_PATH" \
        --det_path "$DET_PATH" \
        --out_video "$OUT_VIDEO" \
        --start_frame "$START_FRAME" \
        --end_frame "$END_FRAME" \
        --sensor_id "$SENSOR_ID" \

    # Post-process the videos with ffmpeg
    for vid_path in "$OUT_VIDEO" "${OUT_VIDEO//.mp4/_bag_only.mp4}"; do
        # Generated mp4s never run correctly, so we need to reprocess them
        ffmpeg -i "$vid_path" -c:v libx264 -c:a aac "${vid_path%.*}_temp.mp4" -y 2>/dev/null
        mv "${vid_path%.*}_temp.mp4" "$vid_path"
        echo "Post-processed $vid_path"
    done

    echo "Finished example $EXAMPLE_NUM"
    echo "========================================"
    echo
done

echo "All examples completed!" 