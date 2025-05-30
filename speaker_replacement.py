import pandas as pd
import numpy as np
import re
import os
from collections import defaultdict

class BatchSpeakerAnalyzer:
    def __init__(self):
        self.stats = {
            "valid_utterances": 0,
            "invalid_utterances": 0,
            "simultaneous_speech_segments": 0
        }
        self.global_stats = {
            "total_videos": 0,
            "processed_videos": 0,
            "failed_videos": 0,
            "total_utterances": 0,
            "total_correct": 0
        }
    
    def ensure_path_exists(self, file_path):
        """检查文件路径的目录是否存在，不存在则创建"""
        try:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"✅ 创建目录: {directory}")
            return True
        except Exception as e:
            print(f"❌ 创建目录失败: {e}")
            return False
    
    def get_unique_videos(self, csv_path):
        """从CSV文件中获取所有唯一的视频名称"""
        try:
            df = pd.read_csv(csv_path)
            if 'video_id' not in df.columns:
                print(f"❌ CSV文件中没有找到'video_id'列")
                return []
            
            unique_videos = df['video_id'].unique().tolist()
            print(f"📹 找到 {len(unique_videos)} 个唯一视频")
            return unique_videos
            
        except Exception as e:
            print(f"❌ 读取CSV文件失败: {e}")
            return []
    
    def find_transcript_file(self, video_name, transcript_base_dir):
        """根据视频名称查找对应的transcript文件"""
        # 将.mp4改为.txt
        if video_name.endswith('.mp4'):
            transcript_name = video_name.replace('.mp4', '.txt')
        else:
            transcript_name = video_name + '.txt'
        
        transcript_path = os.path.join(transcript_base_dir, transcript_name)
        
        if os.path.exists(transcript_path):
            return transcript_path
        else:
            print(f"⚠️  未找到transcript文件: {transcript_path}")
            return None
    
    def time_to_seconds(self, time_str):
        """Convert time string (mm:ss) to seconds"""
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    
    def parse_entity_id(self, entity_id):
        """解析entity_id获取start_time, end_time, player_id"""
        parts = entity_id.split('_')
        
        if len(parts) < 3:
            raise ValueError(f"Invalid entity_id format: {entity_id}")
        
        try:
            player_id = int(parts[-1])
            end_time_str = parts[-2]
            start_time_str = parts[-3]
            
            if '.' in start_time_str:
                start_time = int(float(start_time_str))
            else:
                start_time = int(start_time_str)
                
            if '.' in end_time_str:
                end_time = int(float(end_time_str))
            else:
                end_time = int(end_time_str)
            
            return start_time, end_time, player_id
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid entity_id format: {entity_id}") from e
    
    def parse_transcript(self, transcript_path):
        """解析transcript文件"""
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        utterances = []
        timestamp_to_players = {}
        
        for i, line in enumerate(lines):
            player_match = re.match(r'\[Player(\d+)\]\s*\((\d+):(\d+)\):', line)
            if player_match:
                player_id = int(player_match.group(1))
                minutes = int(player_match.group(2))
                seconds = int(player_match.group(3))
                start_time = minutes * 60 + seconds
                start_time_str = f"{minutes:02d}:{seconds:02d}"
                
                if start_time not in timestamp_to_players:
                    timestamp_to_players[start_time] = []
                timestamp_to_players[start_time].append({
                    "player_id": player_id,
                    "line_idx": i,
                    "start_time_str": start_time_str
                })
        
        sorted_timestamps = sorted(timestamp_to_players.keys())
        
        for i, timestamp in enumerate(sorted_timestamps):
            players = timestamp_to_players[timestamp]
            
            end_time = None
            if i < len(sorted_timestamps) - 1:
                next_timestamp = sorted_timestamps[i+1]
                if next_timestamp > timestamp:
                    end_time = next_timestamp
            
            if end_time is None:
                if i == len(sorted_timestamps) - 1:
                    end_time = timestamp + 10
                else:
                    end_time = timestamp + 5
            
            if end_time is None or timestamp >= end_time:
                self.stats["invalid_utterances"] += len(players)
                continue
            
            if len(players) > 1:
                self.stats["simultaneous_speech_segments"] += 1
            
            for player_info in players:
                utterances.append({
                    "player_id": player_info["player_id"],
                    "start_time": timestamp,
                    "end_time": end_time,
                    "start_time_str": player_info["start_time_str"],
                    "is_simultaneous": len(players) > 1
                })
                self.stats["valid_utterances"] += 1
        
        return utterances
    
    def predict_speakers(self, stats_df, score_method='max', top_k=3):
        """为每个时间段预测说话者"""
        predictions = []
        sort_column = 'max_score' if score_method == 'max' else 'avg_score'
        
        for time_segment, group in stats_df.groupby('time_segment'):
            group_sorted = group.sort_values(sort_column, ascending=False)
            top_players = group_sorted.head(top_k)
            
            prediction = {
                'start_time': int(group_sorted.iloc[0]['start_time']),
                'end_time': int(group_sorted.iloc[0]['end_time']),
                'predicted_players': [],
                'predicted_player_ids': [],
                'score_method': score_method,
                'all_players': {}
            }
            
            for _, row in top_players.iterrows():
                primary_score = float(row[sort_column])
                
                player_info = {
                    'player_id': int(row['player_id']),
                    'max_score': float(row['max_score']),
                    'avg_score': float(row['avg_score']),
                    'primary_score': primary_score
                }
                prediction['predicted_players'].append(player_info)
                prediction['predicted_player_ids'].append(int(row['player_id']))
            
            for _, row in group_sorted.iterrows():
                prediction['all_players'][int(row['player_id'])] = {
                    'max_score': float(row['max_score']),
                    'avg_score': float(row['avg_score']),
                    'frame_count': int(row['frame_count'])
                }
            
            predictions.append(prediction)
        
        predictions.sort(key=lambda x: x['start_time'])
        return predictions
    
    def calculate_player_statistics(self, df_video):
        """计算单个视频的player统计信息"""
        df_video = df_video.copy()
        
        df_video[['start_time', 'end_time', 'player_id']] = df_video['entity_id'].apply(
            lambda x: pd.Series(self.parse_entity_id(x))
        )
        
        df_video['time_segment'] = df_video['start_time'].astype(str) + '_' + df_video['end_time'].astype(str)
        
        stats = df_video.groupby(['time_segment', 'start_time', 'end_time', 'player_id'])['score'].agg([
            'max', 'mean', 'count'
        ]).reset_index()
        
        stats.columns = ['time_segment', 'start_time', 'end_time', 'player_id', 'max_score', 'avg_score', 'frame_count']
        
        return stats
    
    def calculate_utterance_level_accuracy(self, predictions, original_txt_path):
        """计算utterance级别的准确率"""
        with open(original_txt_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()
        
        original_utterances = []
        for line in original_lines:
            player_match = re.match(r'\[Player(\d+)\]\s*\((\d+):(\d+)\):(.*)', line)
            if player_match:
                original_player = int(player_match.group(1))
                minutes = int(player_match.group(2))
                seconds = int(player_match.group(3))
                timestamp = minutes * 60 + seconds
                content = player_match.group(4)
                
                original_utterances.append({
                    'timestamp': timestamp,
                    'original_player': original_player,
                    'minutes': minutes,
                    'seconds': seconds,
                    'content': content.strip(),
                    'time_str': f"{minutes:02d}:{seconds:02d}"
                })
        
        utterance_results = []
        correct_count = 0
        total_count = len(original_utterances)
        
        for utterance in original_utterances:
            predicted_player = None
            prediction_score = 0.0
            
            for pred in predictions:
                if pred['start_time'] <= utterance['timestamp'] < pred['end_time']:
                    if pred['predicted_player_ids']:
                        predicted_player = pred['predicted_player_ids'][0]
                        prediction_score = pred['predicted_players'][0]['primary_score']
                    break
            
            if predicted_player is None:
                predicted_player = utterance['original_player']
                prediction_score = 0.0
            
            is_correct = predicted_player == utterance['original_player']
            if is_correct:
                correct_count += 1
            
            utterance_results.append({
                'timestamp': utterance['timestamp'],
                'time_str': utterance['time_str'],
                'original_player': utterance['original_player'],
                'predicted_player': predicted_player,
                'prediction_score': prediction_score,
                'is_correct': is_correct,
                'content_preview': utterance['content'][:30] + '...' if len(utterance['content']) > 30 else utterance['content']
            })
        
        overall_accuracy = correct_count / total_count if total_count > 0 else 0
        
        return {
            'utterance_results': utterance_results,
            'overall_accuracy': overall_accuracy,
            'correct_count': correct_count,
            'total_count': total_count
        }
    
    def generate_utterance_replacement_txt(self, accuracy_data, original_txt_path, output_path):
        """生成替换的txt文件"""
        results = accuracy_data['utterance_results']
        
        with open(original_txt_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()
        
        modified_lines = []
        utterance_index = 0
        
        for line in original_lines:
            player_match = re.match(r'\[Player(\d+)\]\s*\((\d+):(\d+)\):(.*)', line)
            
            if player_match and utterance_index < len(results):
                result = results[utterance_index]
                minutes = int(player_match.group(2))
                seconds = int(player_match.group(3))
                content = player_match.group(4)
                
                time_str = f"{minutes:02d}:{seconds:02d}"
                new_line = f"[Player{result['predicted_player']}] ({time_str}):{content}"
                
                if not new_line.endswith('\n'):
                    new_line += '\n'
                
                modified_lines.append(new_line)
                utterance_index += 1
            else:
                modified_lines.append(line)
        
        # 确保输出目录存在
        self.ensure_path_exists(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in modified_lines:
                f.write(line)
        
        return accuracy_data['overall_accuracy']
    
    def process_single_video(self, video_name, csv_data, transcript_base_dir, output_base_dir, score_method='max'):
        """处理单个视频"""
        print(f"\n🎬 处理视频: {video_name}")
        
        try:
            # 1. 查找transcript文件
            transcript_path = self.find_transcript_file(video_name, transcript_base_dir)
            if not transcript_path:
                print(f"❌ 跳过视频 {video_name}: 找不到transcript文件")
                return None
            
            # 2. 过滤该视频的数据
            video_data = csv_data[csv_data['video_id'] == video_name].copy()
            if len(video_data) == 0:
                print(f"❌ 跳过视频 {video_name}: CSV中没有数据")
                return None
            
            print(f"   📊 数据行数: {len(video_data)}")
            
            # 3. 计算统计信息
            stats_df = self.calculate_player_statistics(video_data)
            print(f"   📈 统计完成: {len(stats_df)} 个player-时间段组合")
            
            # 4. 生成预测
            predictions = self.predict_speakers(stats_df, score_method=score_method)
            print(f"   🔮 预测完成: {len(predictions)} 个时间段")
            
            # 5. 计算utterance级别准确率
            accuracy_data = self.calculate_utterance_level_accuracy(predictions, transcript_path)
            print(f"   📊 Utterance准确率: {accuracy_data['overall_accuracy']:.2%}")
            
            # 6. 生成输出文件
            if video_name.endswith('.mp4'):
                output_filename = video_name.replace('.mp4', '.txt')
            else:
                output_filename = video_name + '.txt'
            
            output_path = os.path.join(output_base_dir, output_filename)
            final_accuracy = self.generate_utterance_replacement_txt(accuracy_data, transcript_path, output_path)
            
            print(f"   💾 输出文件: {output_path}")
            print(f"   ✅ 处理完成: {accuracy_data['correct_count']}/{accuracy_data['total_count']} utterances正确")
            
            return {
                'video_name': video_name,
                'accuracy': final_accuracy,
                'correct_count': accuracy_data['correct_count'],
                'total_count': accuracy_data['total_count'],
                'output_path': output_path
            }
            
        except Exception as e:
            print(f"❌ 处理视频 {video_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_all_videos(self, csv_path, transcript_base_dir, output_base_dir, score_method='max'):
        """批量处理所有视频"""
        print("🚀 开始批量处理视频...")
        print(f"   📁 CSV文件: {csv_path}")
        print(f"   📁 Transcript目录: {transcript_base_dir}")
        print(f"   📁 输出目录: {output_base_dir}")
        print(f"   🎯 评分方法: {score_method}")
        
        # 确保输出目录存在
        self.ensure_path_exists(os.path.join(output_base_dir, "dummy.txt"))
        
        # 读取CSV数据
        try:
            csv_data = pd.read_csv(csv_path)
            print(f"   📊 CSV数据加载完成: {len(csv_data)} 行")
        except Exception as e:
            print(f"❌ 读取CSV文件失败: {e}")
            return
        
        # 获取所有视频
        unique_videos = self.get_unique_videos(csv_path)
        if not unique_videos:
            print("❌ 没有找到视频")
            return
        
        self.global_stats["total_videos"] = len(unique_videos)
        
        # 处理每个视频
        results = []
        for i, video_name in enumerate(unique_videos, 1):
            print(f"\n{'='*60}")
            print(f"进度: {i}/{len(unique_videos)} - {video_name}")
            print(f"{'='*60}")
            
            result = self.process_single_video(
                video_name, csv_data, transcript_base_dir, output_base_dir, score_method
            )
            
            if result:
                results.append(result)
                self.global_stats["processed_videos"] += 1
                self.global_stats["total_utterances"] += result['total_count']
                self.global_stats["total_correct"] += result['correct_count']
            else:
                self.global_stats["failed_videos"] += 1
        
        # 打印总结
        self.print_batch_summary(results, score_method)
        
        # 保存结果到CSV
        self.save_batch_results(results, output_base_dir, score_method)
    
    def print_batch_summary(self, results, score_method):
        """打印批量处理总结"""
        print(f"\n" + "="*80)
        print(f"批量处理完成总结 ({score_method}方法)")
        print("="*80)
        
        print(f"\n📊 处理统计:")
        print(f"   总视频数: {self.global_stats['total_videos']}")
        print(f"   成功处理: {self.global_stats['processed_videos']}")
        print(f"   处理失败: {self.global_stats['failed_videos']}")
        print(f"   成功率: {self.global_stats['processed_videos']/self.global_stats['total_videos']:.1%}")
        
        if results:
            # 计算总体准确率
            total_utterances = self.global_stats['total_utterances']
            total_correct = self.global_stats['total_correct']
            overall_accuracy = total_correct / total_utterances if total_utterances > 0 else 0
            
            print(f"\n📈 准确率统计:")
            print(f"   总utterances: {total_utterances}")
            print(f"   总正确数: {total_correct}")
            print(f"   **总体utterance准确率: {overall_accuracy:.2%}**")
            
            # 各视频准确率分布
            accuracies = [r['accuracy'] for r in results]
            print(f"\n📊 各视频准确率分布:")
            print(f"   最高准确率: {max(accuracies):.2%}")
            print(f"   最低准确率: {min(accuracies):.2%}")
            print(f"   平均准确率: {np.mean(accuracies):.2%}")
            print(f"   准确率标准差: {np.std(accuracies):.2%}")
            
            # 显示前5和后5的视频
            results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
            
            print(f"\n🏆 准确率最高的5个视频:")
            for i, result in enumerate(results_sorted[:5], 1):
                print(f"   {i}. {result['video_name']}: {result['accuracy']:.2%} "
                      f"({result['correct_count']}/{result['total_count']})")
            
            print(f"\n📉 准确率最低的5个视频:")
            for i, result in enumerate(results_sorted[-5:], 1):
                print(f"   {i}. {result['video_name']}: {result['accuracy']:.2%} "
                      f"({result['correct_count']}/{result['total_count']})")
    
    def save_batch_results(self, results, output_base_dir, score_method):
        """保存批量处理结果到CSV"""
        if not results:
            return
        
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(output_base_dir, f"batch_results_{score_method}.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"\n💾 批量结果已保存到: {results_csv_path}")

def main():
    """主函数"""
    # 配置路径
    csv_path = "/mnt/data2/datasets/xpeng/mmsi/ego4d_asd_test/csv/val_res.csv"
    transcript_base_dir = "/mnt/data2/datasets/xpeng/mmsi/ego4d/transcripts/anonymized/"
    output_base_dir = "/mnt/data2/datasets/xpeng/mmsi/ego4d_asd_test/transcripts_r"
    
    # 创建分析器
    analyzer = BatchSpeakerAnalyzer()
    
    # 选择评分方法 ('max' 或 'avg')
    score_method = 'avg'  # 可以改为 'max' 来测试不同方法
    
    # 批量处理
    analyzer.process_all_videos(csv_path, transcript_base_dir, output_base_dir, score_method)

if __name__ == "__main__":
    main()