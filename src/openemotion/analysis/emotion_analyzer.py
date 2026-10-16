from collections import Counter, defaultdict
from typing import List, Dict, Any
from openemotion.config import CLASS_NAMES


class EmotionAnalyzer:
    """Tổng hợp kết quả nhận diện cảm xúc theo thời gian thành báo cáo chi tiết."""

    @staticmethod
    def analyze(raw_results: List[Dict[str, Any]], duration: float = 0.0) -> Dict[str, Any]:
        """
        Phân tích danh sách raw results để tạo Emotion Distribution, Timeline và Summary.
        """
        valid_results = [r for r in raw_results if r.get("status") != "no_face_detected" and "emotion" in r]

        if not valid_results:
            empty_dist = {emotion: 0.0 for emotion in CLASS_NAMES}
            return {
                "duration": duration,
                "participants": 0,
                "summary": {"dominant_emotion": "neutral"},
                "emotion_distribution": empty_dist,
                "timeline": [],
            }

        # 1. Tính số khuôn mặt trung bình per timestamp (participants estimation)
        timestamp_face_counts = defaultdict(int)
        for r in valid_results:
            timestamp_face_counts[r["timestamp"]] += 1

        avg_participants = (
            round(sum(timestamp_face_counts.values()) / len(timestamp_face_counts))
            if timestamp_face_counts else 0
        )

        # 2. Emotion Distribution
        emotion_counts = Counter(r["emotion"] for r in valid_results)
        total_valid = len(valid_results)
        distribution = {}
        for emotion in CLASS_NAMES:
            distribution[emotion] = round(emotion_counts.get(emotion, 0) / total_valid, 4)

        # 3. Dominant Emotion
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral"

        # 4. Timeline generation (nhóm theo timestamp -> cảm xúc phổ biến nhất tại mốc đó)
        timestamp_groups = defaultdict(list)
        for r in valid_results:
            timestamp_groups[r["timestamp"]].append(r["emotion"])

        timeline = []
        for ts in sorted(timestamp_groups.keys()):
            most_common = Counter(timestamp_groups[ts]).most_common(1)[0][0]
            timeline.append({
                "timestamp": ts,
                "emotion": most_common,
            })

        return {
            "duration": duration,
            "participants": max(1, avg_participants),
            "summary": {
                "dominant_emotion": dominant_emotion,
            },
            "emotion_distribution": distribution,
            "timeline": timeline,
        }
