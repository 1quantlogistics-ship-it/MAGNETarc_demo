import { useState, useEffect } from 'react';
import Card from '../components/shared/Card';
import Badge from '../components/shared/Badge';
import { generateMockDesign } from '../utils/mockData';
import { Search, Filter } from 'lucide-react';

export default function DesignHistory() {
  const [designs, setDesigns] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');

  useEffect(() => {
    const mockDesigns = Array.from({ length: 50 }, (_, i) => generateMockDesign(i));
    setDesigns(mockDesigns);
  }, []);

  const filteredDesigns = designs.filter((design) => {
    const matchesSearch = design.design_id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'all' || design.hypothesis_type === filterType;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold gradient-text mb-2">Design History</h1>
        <p className="text-gray-400">Complete archive of generated designs</p>
      </div>

      {/* Filters */}
      <Card>
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search by design ID..."
              className="input pl-10"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <select
            className="input w-48"
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
          >
            <option value="all">All Types</option>
            <option value="exploration">Exploration</option>
            <option value="exploitation">Exploitation</option>
            <option value="novelty">Novelty</option>
            <option value="optimization">Optimization</option>
          </select>
        </div>
      </Card>

      {/* Table */}
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-navy-700">
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Design ID</th>
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Timestamp</th>
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Type</th>
                <th className="text-right py-3 px-4 text-gray-400 font-semibold">Overall Score</th>
                <th className="text-right py-3 px-4 text-gray-400 font-semibold">Stability</th>
                <th className="text-right py-3 px-4 text-gray-400 font-semibold">Speed</th>
                <th className="text-center py-3 px-4 text-gray-400 font-semibold">Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredDesigns.map((design) => (
                <tr
                  key={design.design_id}
                  className="border-b border-navy-700/50 hover:bg-navy-700/30 cursor-pointer transition-colors"
                >
                  <td className="py-3 px-4 font-mono text-cyan">{design.design_id}</td>
                  <td className="py-3 px-4 text-gray-400">
                    {new Date(design.timestamp).toLocaleString()}
                  </td>
                  <td className="py-3 px-4">
                    <Badge variant="info">{design.hypothesis_type}</Badge>
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-cyan">
                    {design.performance.overall.toFixed(1)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-gray-400">
                    {design.performance.stability.toFixed(1)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-gray-400">
                    {design.performance.speed.toFixed(1)}
                  </td>
                  <td className="py-3 px-4 text-center">
                    {design.isBreakthrough ? (
                      <span className="text-success" title="Breakthrough">⭐</span>
                    ) : (
                      <span className="text-gray-600">✓</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 flex items-center justify-between text-sm text-gray-400">
          <p>Showing {filteredDesigns.length} of {designs.length} designs</p>
          <div className="flex gap-2">
            <button className="px-3 py-1 rounded hover:bg-navy-700/50">Previous</button>
            <button className="px-3 py-1 rounded bg-cyan text-navy-900">1</button>
            <button className="px-3 py-1 rounded hover:bg-navy-700/50">2</button>
            <button className="px-3 py-1 rounded hover:bg-navy-700/50">3</button>
            <button className="px-3 py-1 rounded hover:bg-navy-700/50">Next</button>
          </div>
        </div>
      </Card>
    </div>
  );
}
