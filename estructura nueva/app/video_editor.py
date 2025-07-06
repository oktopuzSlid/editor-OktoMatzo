"""
Módulo principal del editor de video.
Integra YOLO para detección de objetos y aplicación de efectos.
"""
import cv2
import numpy as np
import multiprocessing as mp
from .models.yolo.loader import get_yolo_model
from analysis.object_analyzer import analyze_objects

class VideoEditor:
    """Editor de video con capacidades de detección de objetos YOLO."""
    
    def __init__(self, rtsp_url):
        """
        Inicializa el editor de video.
        
        Args:
            rtsp_url (str): URL del stream RTSP
        """
        self.rtsp_url = rtsp_url
        self.model = get_yolo_model()
        self.object_effects = {}
        self.analysis_process = None
        self.result_queue = mp.Queue()

    def apply_effects(self, frame):
        """
        Aplica efectos a los objetos detectados en el frame.
        
        Args:
            frame (numpy.ndarray): Frame de video a procesar
            
        Returns:
            numpy.ndarray: Frame procesado con efectos aplicados
        """
        processed_frame = frame.copy()
        
        # Detección de objetos con YOLO
        results = self.model(frame, verbose=False)[0]
        boxes = results.boxes
        masks = results.masks.data.cpu().numpy() if results.masks is not None else []
        
        # Procesar cada objeto detectado
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            class_name = self.model.model.names[cls_id]
            
            # Obtener máscara del objeto
            mask = self._get_object_mask(i, masks, box, frame.shape)
            
            # Aplicar efectos si están configurados para esta clase
            if class_name in self.object_effects:
                effect = self.object_effects[class_name]
                processed_frame = self._apply_single_effect(
                    processed_frame, 
                    mask, 
                    effect
                )
        
        return processed_frame

    def _get_object_mask(self, idx, masks, box, frame_shape):
        """Genera máscara binaria para un objeto detectado."""
        if idx < len(masks):
            mask_raw = masks[idx]
            mask_resized = cv2.resize(mask_raw, (frame_shape[1], frame_shape[0]))
            return (mask_resized > 0.5).astype(np.uint8)
        else:
            mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            mask[y1:y2, x1:x2] = 1
            return mask

    def _apply_single_effect(self, frame, mask, effect):
        """Aplica un efecto específico a una región del frame."""
        if effect['action'] == 'eliminar':
            frame[mask == 1] = 0
        elif effect['action'] == 'color':
            color = np.array(effect['color'], dtype=np.uint8)
            color_layer = np.zeros_like(frame, dtype=np.uint8)
            color_layer[:] = color
            frame = np.where(mask[:, :, None] == 1, color_layer, frame)
        elif effect['action'] == 'resaltar':
            contours, _ = cv2.findContours(
                mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)
        return frame

    def run(self):
        """Ejecuta el bucle principal del editor de video."""
        # Inicializar lector de cámara
        camera = RTSPStreamReader(self.rtsp_url)
        camera.start()
        
        try:
            while True:
                frame = camera.get_latest_frame()
                if frame is None:
                    continue
                
                # Procesar frame
                processed_frame = self.apply_effects(frame)
                
                # Mostrar estado y frame
                self._display_frame(processed_frame)
                
                # Manejo de teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord("d"):
                    self._start_analysis(frame)
                elif key == ord("q"):
                    break
                
                # Manejar resultados del análisis
                self._handle_analysis_results()
                
        finally:
            camera.stop()
            cv2.destroyAllWindows()
            if self.analysis_process and self.analysis_process.is_alive():
                self.analysis_process.terminate()

    def _display_frame(self, frame):
        """Muestra el frame procesado con información de estado."""
        status = "Analizando..." if self.analysis_process and self.analysis_process.is_alive() else "Listo"
        cv2.putText(frame, f"Estado: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Editor de Video con YOLO", frame)

    def _start_analysis(self, frame):
        """Inicia el análisis de objetos en un proceso separado."""
        if self.analysis_process is None or not self.analysis_process.is_alive():
            print("\n🔍 Iniciando análisis...")
            self.analysis_process = mp.Process(
                target=analyze_objects,
                args=(frame.copy(), self.result_queue)
            )
            self.analysis_process.start()

    def _handle_analysis_results(self):
        """Procesa los resultados del análisis de objetos."""
        if self.analysis_process and not self.result_queue.empty():
            objects_info = self.result_queue.get()
            self.analysis_process.join()
            self.analysis_process = None
            
            if objects_info:
                print("\n✅ Análisis completado. Resultados:")
                for obj_info in objects_info:
                    print(f"\n📦 Objeto: {obj_info['class_name']} — {obj_info['caption']}")
                    self._configure_object_effect(obj_info['class_name'])

    def _configure_object_effect(self, class_name):
        """Configura efectos para una clase de objeto."""
        action = input(f"¿Qué quieres hacer con los objetos '{class_name}'? (eliminar/color/resaltar/nada): ").strip().lower()
        
        if action in ["eliminar", "color", "resaltar"]:
            effect = {'action': action}
            
            if action == "color":
                color_input = input("Color (R,G,B ej: 0,255,0): ").strip()
                try:
                    r, g, b = map(int, color_input.split(","))
                    effect['color'] = (b, g, r)  # OpenCV usa BGR
                except:
                    print("Color inválido. Usando verde por defecto.")
                    effect['color'] = (0, 255, 0)
            
            self.object_effects[class_name] = effect
            print(f"Efecto '{action}' aplicado a {class_name}")