export default function Badge({ children, variant = 'info', className = '' }) {
  const variants = {
    success: 'badge-success',
    warning: 'badge-warning',
    alert: 'badge-alert',
    info: 'badge-info',
  };

  return (
    <span className={`badge ${variants[variant] || variants.info} ${className}`}>
      {children}
    </span>
  );
}
