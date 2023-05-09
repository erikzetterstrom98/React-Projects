import {ReactNode} from 'react'

interface Props {
    children: ReactNode;
    onClick: React.MouseEventHandler<HTMLButtonElement>;
}

export const Button = ({ children, onClick}: Props) => {
  return (
    <button 
        type='button'
        className='btn btn-primary cool-button'
        onClick={onClick}
    >
            {children}
    </button>
  )
}
