class WeightCalculator:
    def __init__(self, calibration_weight_kg=500.0, base_calibration_area_m2=2.0):
        """
        Calculadora de peso baseada na área top-down.
        Variáveis de calibração para serem ajustadas após testes empíricos.
        """
        self.calibration_weight_kg = calibration_weight_kg
        self.base_calibration_area_m2 = base_calibration_area_m2
        
        # Densidade do gado visto de cima (kg por metro quadrado de pixel)
        self.density_kg_per_m2 = self.calibration_weight_kg / self.base_calibration_area_m2

    def calculate_weight(self, pixel_area, scale_cm_per_pixel):
        """
        Converte a área em pixels (segmentação) para uma estimativa de peso em Kg.
        """
        if scale_cm_per_pixel is None or scale_cm_per_pixel <= 0:
            return 0.0
        
        # Área real equivalente a 1 pixel na tela
        pixel_area_cm2 = scale_cm_per_pixel ** 2
        
        # Converter os pixels totais da máscara para a área real do animal
        real_area_cm2 = pixel_area * pixel_area_cm2
        real_area_m2 = real_area_cm2 / 10000.0
        
        # Obter a estimativa linear de peso
        estimated_weight_kg = real_area_m2 * self.density_kg_per_m2
        
        return estimated_weight_kg
