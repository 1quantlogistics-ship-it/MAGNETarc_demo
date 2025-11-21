import { useState, useEffect } from 'react';
import Card from '../components/shared/Card';
import Badge from '../components/shared/Badge';
import { generateMockPrinciple } from '../utils/mockData';
import { BookOpen, TrendingUp, AlertTriangle } from 'lucide-react';

export default function KnowledgeBase() {
  const [principles, setPrinciples] = useState([]);
  const [selectedPrinciple, setSelectedPrinciple] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('all');

  useEffect(() => {
    const mockPrinciples = Array.from({ length: 20 }, () => generateMockPrinciple());
    setPrinciples(mockPrinciples);
    setSelectedPrinciple(mockPrinciples[0]);
  }, []);

  const categories = ['all', 'Stability', 'Speed', 'Efficiency', 'Trade-offs', 'Failures'];

  const filteredPrinciples = selectedCategory === 'all'
    ? principles
    : principles.filter(p => p.category === selectedCategory);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold gradient-text mb-2">Knowledge Base</h1>
        <p className="text-gray-400">Design principles discovered through autonomous research</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Panel - Principles List */}
        <Card title="Design Principles" className="lg:col-span-1">
          {/* Category Filter */}
          <div className="flex flex-wrap gap-2 mb-4">
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1 rounded-lg text-sm capitalize ${
                  selectedCategory === cat
                    ? 'bg-cyan text-navy-900'
                    : 'bg-navy-700 hover:bg-navy-600'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Principles List */}
          <div className="space-y-2 max-h-[600px] overflow-y-auto">
            {filteredPrinciples.map((principle, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedPrinciple(principle)}
                className={`glass-card p-3 rounded-lg cursor-pointer transition-all ${
                  selectedPrinciple === principle
                    ? 'bg-cyan/20 border-l-2 border-cyan'
                    : 'hover:bg-navy-700/30'
                }`}
              >
                <div className="flex items-start gap-2 mb-2">
                  <BookOpen className="w-4 h-4 text-cyan mt-1 flex-shrink-0" />
                  <div className="flex-1">
                    <Badge variant="info" className="text-xs mb-2">
                      {principle.category}
                    </Badge>
                    <p className="text-sm line-clamp-2">{principle.statement}</p>
                  </div>
                </div>
                <div className="flex items-center justify-between text-xs mt-2">
                  <span className="text-gray-400">
                    Confidence: {(principle.confidence * 100).toFixed(0)}%
                  </span>
                  <span className="text-gray-400">
                    {new Date(principle.discovered).toLocaleDateString()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Right Panel - Principle Details */}
        <Card title="Principle Details" className="lg:col-span-2">
          {selectedPrinciple ? (
            <div className="space-y-6">
              {/* Statement */}
              <div>
                <h3 className="text-2xl mb-4">{selectedPrinciple.statement}</h3>
                <div className="flex gap-3">
                  <Badge variant="info">{selectedPrinciple.category}</Badge>
                  <Badge variant="success">
                    Confidence: {(selectedPrinciple.confidence * 100).toFixed(0)}%
                  </Badge>
                  <Badge variant="warning">
                    {selectedPrinciple.evidence} supporting designs
                  </Badge>
                </div>
              </div>

              {/* Confidence Visualization */}
              <div>
                <p className="text-sm text-gray-400 mb-2">Confidence Score</p>
                <div className="flex items-center gap-4">
                  <div className="flex-1 h-4 bg-navy-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan to-success"
                      style={{ width: `${selectedPrinciple.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-2xl font-bold text-cyan">
                    {(selectedPrinciple.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Evidence */}
              <div>
                <h4 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-success" />
                  Supporting Evidence
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  {Array.from({ length: 6 }, (_, i) => (
                    <div key={i} className="glass-card p-3 rounded-lg">
                      <p className="text-xs font-mono text-gray-400 mb-2">
                        design_{Date.now() - i * 1000}
                      </p>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Score:</span>
                        <span className="font-mono text-success">
                          {(75 + Math.random() * 20).toFixed(1)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Actionable Insights */}
              <div>
                <h4 className="text-lg font-semibold mb-3">Actionable Insights</h4>
                <div className="glass-card p-4 rounded-lg">
                  <p className="text-sm">
                    Future designs should prioritize this parameter range to achieve optimal
                    performance in the {selectedPrinciple.category.toLowerCase()} domain.
                    Consider combining with related principles for compounded benefits.
                  </p>
                </div>
              </div>

              {/* Related Principles */}
              <div>
                <h4 className="text-lg font-semibold mb-3">Related Principles</h4>
                <div className="flex flex-wrap gap-2">
                  {principles.slice(0, 3).map((p, idx) => (
                    <button
                      key={idx}
                      onClick={() => setSelectedPrinciple(p)}
                      className="px-3 py-2 glass-card rounded-lg hover:bg-navy-700/50 text-sm text-left"
                    >
                      {p.statement.slice(0, 60)}...
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-400">
              <p>Select a principle to view details</p>
            </div>
          )}
        </Card>
      </div>

      {/* Failure Patterns Section */}
      <Card title="Failure Patterns">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="glass-card p-4 rounded-lg border-l-2 border-alert">
            <div className="flex items-start gap-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-alert" />
              <h4 className="font-semibold">Excessive Draft</h4>
            </div>
            <p className="text-sm text-gray-400 mb-2">
              Designs with draft &gt; 0.8m consistently fail stability requirements
            </p>
            <p className="text-xs text-gray-400">Frequency: 23 occurrences</p>
          </div>

          <div className="glass-card p-4 rounded-lg border-l-2 border-alert">
            <div className="flex items-start gap-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-alert" />
              <h4 className="font-semibold">Narrow Hull Spacing</h4>
            </div>
            <p className="text-sm text-gray-400 mb-2">
              Hull spacing below 1.8m leads to metacentric instability
            </p>
            <p className="text-xs text-gray-400">Frequency: 17 occurrences</p>
          </div>

          <div className="glass-card p-4 rounded-lg border-l-2 border-alert">
            <div className="flex items-start gap-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-alert" />
              <h4 className="font-semibold">Extreme Length Ratios</h4>
            </div>
            <p className="text-sm text-gray-400 mb-2">
              L/B ratios above 12 introduce structural concerns
            </p>
            <p className="text-xs text-gray-400">Frequency: 12 occurrences</p>
          </div>
        </div>
      </Card>
    </div>
  );
}
