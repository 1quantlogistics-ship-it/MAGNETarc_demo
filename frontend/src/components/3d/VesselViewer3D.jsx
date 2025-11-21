import { Suspense, useState } from 'react';
import { Canvas, useLoader } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import LoadingSpinner from '../shared/LoadingSpinner';
import { Eye, EyeOff, Grid as GridIcon } from 'lucide-react';

function VesselMesh({ meshUrl, performanceData }) {
  try {
    const geometry = useLoader(STLLoader, meshUrl);

    const getColor = (score = 0) => {
      if (score >= 80) return '#00ff88';
      if (score >= 60) return '#ffd700';
      return '#ff4444';
    };

    return (
      <mesh geometry={geometry} castShadow receiveShadow>
        <meshPhongMaterial
          color={getColor(performanceData?.overall)}
          shininess={100}
        />
      </mesh>
    );
  } catch (error) {
    console.error('Failed to load mesh:', error);
    return null;
  }
}

function WaterPlane() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.1, 0]} receiveShadow>
      <planeGeometry args={[100, 100]} />
      <meshStandardMaterial
        color="#00a8cc"
        transparent
        opacity={0.3}
        roughness={0.1}
        metalness={0.8}
      />
    </mesh>
  );
}

function Scene({ meshUrl, performanceData, showGrid, showWater }) {
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight
        position={[20, 30, 20]}
        intensity={0.8}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <pointLight position={[-20, 20, -20]} intensity={0.5} />

      {/* Environment */}
      {showWater && <WaterPlane />}
      {showGrid && <Grid args={[50, 50]} cellColor="#16213e" sectionColor="#00d9ff" />}

      {/* Vessel mesh */}
      {meshUrl && <VesselMesh meshUrl={meshUrl} performanceData={performanceData} />}

      {/* Camera and controls */}
      <PerspectiveCamera makeDefault position={[30, 20, 30]} />
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        maxPolarAngle={Math.PI / 2}
        minDistance={10}
        maxDistance={100}
      />
    </>
  );
}

export default function VesselViewer3D({
  meshUrl,
  performanceData,
  showGrid: initialShowGrid = true,
  showWater: initialShowWater = true,
  height = '400px',
}) {
  const [showGrid, setShowGrid] = useState(initialShowGrid);
  const [showWater, setShowWater] = useState(initialShowWater);

  return (
    <div className="relative bg-navy-900 rounded-lg overflow-hidden" style={{ height }}>
      {/* Controls overlay */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        <button
          onClick={() => setShowGrid(!showGrid)}
          className={`p-2 rounded-lg backdrop-blur-xl ${
            showGrid ? 'bg-cyan/20 text-cyan' : 'bg-navy-800/50 text-gray-400'
          }`}
          title="Toggle Grid"
        >
          <GridIcon className="w-5 h-5" />
        </button>
        <button
          onClick={() => setShowWater(!showWater)}
          className={`p-2 rounded-lg backdrop-blur-xl ${
            showWater ? 'bg-cyan/20 text-cyan' : 'bg-navy-800/50 text-gray-400'
          }`}
          title="Toggle Water"
        >
          {showWater ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
        </button>
      </div>

      {/* Performance badge */}
      {performanceData && (
        <div className="absolute top-4 left-4 z-10 glass-card px-3 py-2">
          <div className="text-xs text-gray-400">Overall Score</div>
          <div className="text-2xl font-bold text-cyan">
            {Math.round(performanceData.overall)}
          </div>
        </div>
      )}

      {/* 3D Canvas */}
      <Canvas shadows>
        <Suspense fallback={null}>
          <Scene
            meshUrl={meshUrl}
            performanceData={performanceData}
            showGrid={showGrid}
            showWater={showWater}
          />
        </Suspense>
      </Canvas>

      {/* Loading state */}
      {!meshUrl && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <LoadingSpinner size="lg" />
            <p className="mt-4 text-gray-400">No mesh available</p>
          </div>
        </div>
      )}
    </div>
  );
}
