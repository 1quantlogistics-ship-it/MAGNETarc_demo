import { motion } from 'framer-motion';

export default function PerformanceGauge({ score, label, size = 'md' }) {
  const sizes = {
    sm: { dimension: 80, strokeWidth: 6, fontSize: 'text-lg' },
    md: { dimension: 120, strokeWidth: 8, fontSize: 'text-2xl' },
    lg: { dimension: 160, strokeWidth: 10, fontSize: 'text-3xl' },
  };

  const { dimension, strokeWidth, fontSize } = sizes[size];
  const radius = (dimension - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (score / 100) * circumference;

  const getColor = () => {
    if (score >= 80) return '#00ff88'; // success
    if (score >= 60) return '#ffd700'; // warning
    return '#ff4444'; // alert
  };

  return (
    <div className="flex flex-col items-center">
      <svg width={dimension} height={dimension} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={dimension / 2}
          cy={dimension / 2}
          r={radius}
          stroke="#16213e"
          strokeWidth={strokeWidth}
          fill="none"
        />
        {/* Progress circle */}
        <motion.circle
          cx={dimension / 2}
          cy={dimension / 2}
          r={radius}
          stroke={getColor()}
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
      </svg>
      <div className="mt-2 text-center">
        <div className={`font-bold ${fontSize}`} style={{ color: getColor() }}>
          {Math.round(score)}
        </div>
        {label && <div className="text-sm text-gray-400">{label}</div>}
      </div>
    </div>
  );
}
