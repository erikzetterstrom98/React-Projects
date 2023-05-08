
interface Props {
    strong: string;
    text: string;
    onClick: () => void;
}

export const Alert = ({ strong, text, onClick }: Props) => {
  return (
    <div className="alert alert-warning alert-dismissible fade show" role="alert">
        <strong>{strong}</strong> {text}
        <button type="button" className="btn-close" data-bs-dismiss="alert" aria-label="Close" onClick={onClick}></button>
    </div>
  )
}