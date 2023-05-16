import {useRef, useEffect, ReactNode} from 'react'

interface CanvasProps {
    contextReference: React.MutableRefObject<CanvasRenderingContext2D | null>;
    canvas_height: number;
    canvas_width: number;
    children: ReactNode;
}

export const Canvas = ({contextReference, canvas_height, canvas_width}: CanvasProps) => {

    const canvasReference = useRef<HTMLCanvasElement | null>(null);
    //contextReference = useRef<CanvasRenderingContext2D | null>(null);
    //const contextReference = props.contextReference;
    var isDrawing = false;

    useEffect(
        () => {
            const canvas = canvasReference.current;

            if (!canvas) {
                return;
            }

            canvas.addEventListener('mousedown', startDrawing);
            canvas.addEventListener('mouseup', stopDrawing);
            canvas.addEventListener('mousemove', draw);
            canvas.addEventListener('mouseleave', stopDrawing);

            const context = canvas.getContext('2d');

            if (!context) {
                return;
            }

            contextReference.current = context;
            contextReference.current.lineCap = 'round';
            setPencilStyle('pencil');
        },
        []
    )

    function startDrawing({offsetX, offsetY}: MouseEvent) {
        isDrawing = true;
        contextReference.current?.beginPath();
        contextReference.current?.moveTo(offsetX, offsetY);
        contextReference.current?.lineTo(offsetX, offsetY);
        contextReference.current?.stroke();
    }

    function stopDrawing() {
        isDrawing = false;
        contextReference.current?.closePath();
    }

    function draw({offsetX, offsetY, shiftKey}: MouseEvent) {
        if (!isDrawing || contextReference.current == null) {
            return;
        }

        if (!shiftKey) {
            setPencilStyle('pencil');
        }
        else {
            setPencilStyle('eraser');
        }

        contextReference.current.lineTo(offsetX, offsetY);
        contextReference.current.stroke();
    }

    function setPencilStyle(style: string) {
        if (!contextReference.current) {
            return;
        }
        if (style=='pencil') {
            contextReference.current.strokeStyle = 'black';
            contextReference.current.lineWidth = 5;
        }
        else if (style=='eraser') {
            contextReference.current.strokeStyle = 'white';
            contextReference.current.lineWidth = 25;
        }
        else {
            console.debug('Warning: \'' + style + '\' is not a valid style.');
        }
    }

    return (
        <canvas
            ref={canvasReference}
            height={canvas_height}
            width={canvas_width}
        />
    )
}
