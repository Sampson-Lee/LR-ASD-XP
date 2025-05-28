import os
import shutil
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm
from IPython import embed

def analyze_predictions(ground_truth_file, predictions_file, threshold=0.5, output_file=None):
    """
    Analyze comparison between prediction results and ground truth
    
    Args:
        ground_truth_file: Path to ground truth CSV file
        predictions_file: Path to predictions CSV file
        threshold: Classification threshold, default 0.5
        output_file: Output file name (optional)
        
    Returns:
        tuple: (metrics_dict, result_df) - analysis metrics and detailed results DataFrame
    """
    # Read files
    gt_df = pd.read_csv(ground_truth_file)
    pred_df = pd.read_csv(predictions_file)
    
    # Print column names for debugging
    print("Ground truth columns:", gt_df.columns.tolist())
    print("Predictions columns:", pred_df.columns.tolist())
    
    # Create uid column for merging
    gt_df['uid'] = gt_df['frame_timestamp'].astype(str) + ':' + gt_df['entity_id']
    pred_df['uid'] = pred_df['frame_timestamp'].astype(str) + ':' + pred_df['entity_id']
    
    # Merge data - keep track of important column names
    df = gt_df.merge(pred_df, on='uid', suffixes=('_gt', '_pred'))
    
    # Print merged columns for debugging
    print("Merged DataFrame columns:", df.columns.tolist())
    
    # Get entity_id column with the correct suffix
    entity_id_col = 'entity_id_gt' if 'entity_id_gt' in df.columns else 'entity_id_pred'
    
    # Convert ground truth labels to binary
    df['gt_binary'] = (df['label_gt'] == 'SPEAKING_AUDIBLE').astype(int)
    
    # Convert prediction scores to binary based on threshold
    df['pred_binary'] = (df['score'] >= threshold).astype(int)
    
    # Calculate prediction metrics
    true_pos = ((df['gt_binary'] == 1) & (df['pred_binary'] == 1)).sum()
    false_pos = ((df['gt_binary'] == 0) & (df['pred_binary'] == 1)).sum()
    true_neg = ((df['gt_binary'] == 0) & (df['pred_binary'] == 0)).sum()
    false_neg = ((df['gt_binary'] == 1) & (df['pred_binary'] == 0)).sum()
    
    accuracy = (true_pos + true_neg) / len(df)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print statistics
    print("\n===== Prediction Analysis =====")
    print(f"Total samples: {len(df)}")
    print(f"Speaking (SPEAKING_AUDIBLE) samples in ground truth: {df['gt_binary'].sum()} ({df['gt_binary'].mean()*100:.2f}%)")
    print(f"Not speaking (NOT_SPEAKING) samples in ground truth: {len(df) - df['gt_binary'].sum()} ({(1-df['gt_binary'].mean())*100:.2f}%)")
    print(f"\nClassification metrics (threshold={threshold}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 score: {f1:.4f}")
    print(f"  True positives (TP): {true_pos}")
    print(f"  False positives (FP): {false_pos}")
    print(f"  True negatives (TN): {true_neg}")
    print(f"  False negatives (FN): {false_neg}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(df['gt_binary'], df['pred_binary'])
    
    # Calculate average precision (AP)
    ap = average_precision_score(df['gt_binary'], df['score'])
    print(f"\nAverage Precision (AP): {ap:.4f}")
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, thresholds = precision_recall_curve(df['gt_binary'], df['score'])
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(recall_curve, precision_curve, 'b-', label=f'AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    
    # Plot confusion matrix
    plt.subplot(2, 1, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Speaking', 'Speaking'],
                yticklabels=['Not Speaking', 'Speaking'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (threshold={threshold})')
    
    plt.tight_layout()
    
    # Save or display the figure
    if output_file:
        plt.savefig(output_file)
        print(f"Analysis chart saved to: {output_file}")
    else:
        plt.show()
    
    # Generate detailed CSV file
    result_df = None
    if output_file:
        csv_output = output_file.replace('.png', '.csv')
        
        # Select columns for the result CSV, using the correct column names
        cols_to_keep = [
            'video_id_gt', 
            'frame_timestamp_gt', 
            entity_id_col,  # Use the correct entity_id column 
            'label_gt', 
            'score', 
            'gt_binary', 
            'pred_binary'
        ]
        
        # Keep useful columns
        result_df = df[cols_to_keep].copy()
        
        # Rename entity_id column to make it clearer
        result_df = result_df.rename(columns={entity_id_col: 'entity_id'})
        
        # Add correct prediction column
        result_df['correct'] = (result_df['gt_binary'] == result_df['pred_binary'])
        
        # Ensure proper types for sorting
        if result_df['frame_timestamp_gt'].dtype == 'object':
            # Try to convert frame_timestamp to numeric if needed
            try:
                # If format is like "0.00.jpg", extract the numeric part
                if result_df['frame_timestamp_gt'].str.contains('.jpg').any():
                    result_df['frame_timestamp_numeric'] = result_df['frame_timestamp_gt'].str.replace('.jpg', '').astype(float)
                else:
                    result_df['frame_timestamp_numeric'] = pd.to_numeric(result_df['frame_timestamp_gt'])
                
                # Sort by entity_id, then by frame_timestamp numeric value
                result_df = result_df.sort_values(['entity_id', 'frame_timestamp_numeric'])
                
                # Drop the temporary numeric column
                result_df = result_df.drop('frame_timestamp_numeric', axis=1)
            except:
                # If conversion fails, just sort by the original columns
                result_df = result_df.sort_values(['entity_id', 'frame_timestamp_gt'])
        else:
            # If timestamp is already numeric, sort directly
            result_df = result_df.sort_values(['entity_id', 'frame_timestamp_gt'])
        
        # Save sorted results
        result_df.to_csv(csv_output, index=False)
        print(f"Detailed analysis results saved to: {csv_output}")
        print(f"Results sorted by entity_id and frame_timestamp_gt")
        
        # Print sample of sorted results
        print("\nSample of sorted results (first 5 rows):")
        print(result_df[['entity_id', 'frame_timestamp_gt', 'label_gt', 'score', 'correct']].head())
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap': ap,
        'true_pos': true_pos,
        'false_pos': false_pos,
        'true_neg': true_neg,
        'false_neg': false_neg
    }
    
    return metrics, result_df


def copy_misclassified_images(analysis_df, video_clips_dir, confusion_dir, copy_type='all'):
    """
    Copy misclassified images to a new directory structure
    
    Args:
        analysis_df: DataFrame with analysis results
        video_clips_dir: Path to the original video clips directory with structure:
                         video_id/entity_id/frame_timestamp.jpg
        confusion_dir: Path to the new confusion directory
        copy_type: 'all' for all misclassifications, 'fp' for false positives only, 
                   'fn' for false negatives only
    """
    # Filter based on copy_type
    if copy_type == 'fp':
        # False positives: predicted as speaking but actually not speaking
        df = analysis_df[(analysis_df['pred_binary'] == 1) & (analysis_df['gt_binary'] == 0)]
        print(f"Found {len(df)} false positives")
    elif copy_type == 'fn':
        # False negatives: predicted as not speaking but actually speaking
        df = analysis_df[(analysis_df['pred_binary'] == 0) & (analysis_df['gt_binary'] == 1)]
        print(f"Found {len(df)} false negatives")
    else:  # 'all'
        # All misclassifications
        df = analysis_df[~analysis_df['correct']]
        print(f"Found {len(df)} misclassified images")
    
    # Extract components needed for directory structure
    print("Preparing to copy files...")
    
    # Create the main confusion directory if it doesn't exist
    os.makedirs(confusion_dir, exist_ok=True)
    
    # Counter for files
    copied_count = 0
    not_found_count = 0
    already_exists_count = 0
    
    # Create subdirectories for false positives and false negatives
    fp_dir = os.path.join(confusion_dir, "false_positives")
    fn_dir = os.path.join(confusion_dir, "false_negatives")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)
    
    # Process each misclassified image
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying files"):
        # Extract information needed to locate the file
        video_id = row['video_id_gt']
        frame_timestamp = row['frame_timestamp_gt']
        entity_id = row['entity_id']
        
        # Determine if it's a false positive or false negative
        is_fp = (row['pred_binary'] == 1) and (row['gt_binary'] == 0)
        is_fn = (row['pred_binary'] == 0) and (row['gt_binary'] == 1)
        
        # Use the exact frame_timestamp as the filename - IMPORTANT: preserve original format
        # DO NOT modify the frame_timestamp format (e.g., don't convert 0.80.jpg to 0.8.jpg)
        if isinstance(frame_timestamp, str):
            image_filename = frame_timestamp
            # Make sure it has the .jpg extension
            if not image_filename.endswith('.jpg'):
                image_filename = f"{image_filename}.jpg"
        else:
            # If it's a numeric type, format it to match the original format
            # Assuming format is like 0.XX where XX is always 2 digits
            image_filename = f"{frame_timestamp:.3f}.jpg"
        
        # Print a sample path for verification
        if copied_count == 0 and not_found_count == 0:
            print(f"Sample path: {os.path.join(video_clips_dir, video_id, entity_id, image_filename)}")
        
        # Construct the source path - using video_id/entity_id/frame_timestamp.jpg structure
        source_path = os.path.join(video_clips_dir, video_id, entity_id, image_filename)
        
        # Try alternative filenames if the standard one is not found
        if not os.path.exists(source_path) and isinstance(frame_timestamp, float):
            # Try alternative formats
            alternatives = [
                f"{frame_timestamp:.2f}.jpg",  # 0.80.jpg
                f"{frame_timestamp:g}.jpg",    # 0.8.jpg
                f"{int(frame_timestamp*100)}.jpg"  # 80.jpg
            ]
            
            for alt_filename in alternatives:
                alt_path = os.path.join(video_clips_dir, video_id, entity_id, alt_filename)
                if os.path.exists(alt_path):
                    source_path = alt_path
                    image_filename = alt_filename
                    break
        
        # Determine destination directory (FP or FN)
        if is_fp:
            dest_base_dir = fp_dir
        else:
            dest_base_dir = fn_dir
        
        # Create destination directory structure (video_id/entity_id)
        dest_video_dir = os.path.join(dest_base_dir, video_id)
        dest_entity_dir = os.path.join(dest_video_dir, entity_id)
        os.makedirs(dest_entity_dir, exist_ok=True)
        
        # Construct the destination path
        dest_path = os.path.join(dest_entity_dir, image_filename)
        
        # Copy the file if it exists
        if os.path.exists(source_path):
            if not os.path.exists(dest_path):
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            else:
                already_exists_count += 1
        else:
            print(f"Warning: Source file not found: {source_path}")
            not_found_count += 1
    
    # Summary
    print(f"\nCopying complete!")
    print(f"Files copied: {copied_count}")
    print(f"Files not found: {not_found_count}")
    print(f"Files already existed: {already_exists_count}")
    
    # Create a README file with information about the copied files
    readme_path = os.path.join(confusion_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write("Misclassified Images Summary\n")
        f.write("==========================\n\n")
        f.write(f"Copy type: {copy_type}\n\n")
        f.write(f"Total misclassified images: {len(df)}\n")
        f.write(f"Files copied: {copied_count}\n")
        f.write(f"Files not found: {not_found_count}\n")
        f.write(f"Files already existed: {already_exists_count}\n\n")
        f.write("Directory structure:\n")
        f.write("- false_positives/video_id/entity_id/frame_timestamp.jpg - Predicted as speaking but actually not speaking\n")
        f.write("- false_negatives/video_id/entity_id/frame_timestamp.jpg - Predicted as not speaking but actually speaking\n")
    
    print(f"Created summary in {readme_path}")
    return copied_count
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze prediction results and copy misclassified images')
    parser.add_argument('--ground_truth', default="/mnt/data2/datasets/xpeng/mmsi/ego4d_asdv2/csv/val_orig.csv", 
                        help='Path to ground truth CSV file')
    parser.add_argument('--predictions', default="/mnt/data2/datasets/xpeng/mmsi/ego4d_asdv2/val_res.csv", 
                        help='Path to predictions CSV file')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--output', default="analysis.png", 
                        help='Path to output chart file')
    parser.add_argument('--video_clips', default="/mnt/data2/datasets/xpeng/mmsi/ego4d_asdv2/video_clips/", 
                        help='Path to the original video clips directory. If provided, will copy misclassified images')
    parser.add_argument('--confusion_dir', default="/mnt/data2/datasets/xpeng/mmsi/ego4d_asdv2/confusion/", 
                        help='Path to the new confusion directory for misclassified images')
    parser.add_argument('--copy_type', choices=['all', 'fp', 'fn'], default='all',
                        help='Type of misclassifications to copy (all, fp=false positives, fn=false negatives)')
    
    args = parser.parse_args()
    
    # Print analysis parameters
    print("===== Analysis Parameters =====")
    print(f"Ground truth file: {args.ground_truth}")
    print(f"Predictions file: {args.predictions}")
    print(f"Threshold: {args.threshold}")
    print(f"Output file: {args.output}")
    
    # Run analysis
    metrics, result_df = analyze_predictions(
        args.ground_truth, 
        args.predictions, 
        args.threshold, 
        args.output
    )
    
    # Copy misclassified images if video_clips directory is provided
    if args.video_clips:
        if not os.path.exists(args.video_clips):
            print(f"Error: Video clips directory not found: {args.video_clips}")
        else:
            print("\n===== Copying Misclassified Images =====")
            print(f"Video clips directory: {args.video_clips}")
            print(f"Confusion directory: {args.confusion_dir}")
            print(f"Copy type: {args.copy_type}")
            
            copied = copy_misclassified_images(
                result_df,
                args.video_clips, 
                args.confusion_dir,
                args.copy_type
            )
            
            print(f"Done! Copied {copied} files.")
    else:
        print("\nNo video_clips directory provided. Skipping image copying.")
        print("To copy misclassified images, provide --video_clips parameter")