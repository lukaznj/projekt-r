import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import ipywidgets as widgets
from IPython.display import display, clear_output
import io
from PIL import Image

# Računanje metrika


def calculate_ciede2000(rgb1, rgb2):
    """
    Izračunava prosječnu CIEDE2000 razliku boje između dvije RGB slike.
    Koristi scikit-image.
    """

    # Konverzija RGB → LAB, format koji krositi scikit image
    lab1 = rgb2lab(rgb1)
    lab2 = rgb2lab(rgb2)

    # Izračun CIEDE2000 za svaki piksel
    delta_e = deltaE_ciede2000(lab1, lab2)

    # Vraćanje prosječne vrijednosti
    return np.mean(delta_e)


def evaluate_reconstruction(ref_rgb, recon_rgb, name):
    """Izračunava sve tražene metrike i ispisuje rezultate."""

    # 1. PSNR i SSIM (na luminanciji - Y kanal nakon konverzije u YCbCr)
    ref_ycbcr = cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2YCrCb)
    recon_ycbcr = cv2.cvtColor(recon_rgb, cv2.COLOR_BGR2YCrCb)

    psnr_y = peak_signal_noise_ratio(ref_ycbcr[:, :, 0], recon_ycbcr[:, :, 0])
    ssim_y = structural_similarity(
        ref_ycbcr[:, :, 0],
        recon_ycbcr[:, :, 0],
        data_range=ref_ycbcr[:, :, 0].max() - ref_ycbcr[:, :, 0].min(),
    )

    # 2. PSNR i SSIM po kanalima boje (B, G, R)
    psnr_b = peak_signal_noise_ratio(ref_rgb[:, :, 0], recon_rgb[:, :, 0])
    psnr_g = peak_signal_noise_ratio(ref_rgb[:, :, 1], recon_rgb[:, :, 1])
    psnr_r = peak_signal_noise_ratio(ref_rgb[:, :, 2], recon_rgb[:, :, 2])

    # 3. CIEDE2000 (Prosječna pogreška boje)
    delta_e = calculate_ciede2000(ref_rgb, recon_rgb)

    results = {
        "Metoda": name,
        "PSNR (Y)": psnr_y,
        "SSIM (Y)": ssim_y,
        "PSNR (R)": psnr_r,
        "PSNR (G)": psnr_g,
        "PSNR (B)": psnr_b,
        "CIEDE2000 (ΔE*)": delta_e,
    }
    return results


# Simulacija šuma
def simulate_noise(mosaic, noise_level):
    """
    Simulira Gaussov šum za 'slabu rasvjetu'.
    """
    if noise_level == 0:
        return mosaic

    # np.std() mjeri raspršenost pikselnih vrijednosti i ne koristi se za
    # generiranje šuma pa je zato potrebno ručno izračunati standardne devijacije
    sigma = noise_level * np.max(mosaic)
    noise = np.random.normal(0, sigma, mosaic.shape)
    noisy_mosaic = mosaic.astype(float) + noise

    return np.clip(noisy_mosaic, 0, 255).astype(mosaic.dtype)


# Generiranje mozaika za Quad Bayer
# Na isti način radi i Tetracell
def generate_quad_bayer_mosaic(rgb_img):
    """
    Simulacija Quad Bayer mozaika (2x2 blokovi iste boje).
    Standardni uzorak je 4x4 super-piksel koji sadrži 2x2 R, 2x2 G, 2x2 G, 2x2 B.
    Ovdje implementiramo uzorkovanje gdje se unutar 4x4 bloka (tj. 4x4 piksela)
    svaki 2x2 blok piksela uzorkuje iz iste boje.
    """
    H, W = rgb_img.shape[:2]
    mosaic = np.zeros((H, W), dtype=rgb_img.dtype)

    # Stvaranje prvog 2x2 bloka (Crveni)
    mosaic[0::4, 0::4] = rgb_img[0::4, 0::4, 2]
    mosaic[0::4, 1::4] = rgb_img[0::4, 1::4, 2]
    mosaic[1::4, 0::4] = rgb_img[1::4, 0::4, 2]
    mosaic[1::4, 1::4] = rgb_img[1::4, 1::4, 2]

    # Stvaranje drugog 2x2 bloka (Zeleni)
    mosaic[0::4, 2::4] = rgb_img[0::4, 2::4, 1]
    mosaic[0::4, 3::4] = rgb_img[0::4, 3::4, 1]
    mosaic[1::4, 2::4] = rgb_img[1::4, 2::4, 1]
    mosaic[1::4, 3::4] = rgb_img[1::4, 3::4, 1]

    # Stvaranje trećeg 2x2 bloka (Zeleni)
    mosaic[2::4, 0::4] = rgb_img[2::4, 0::4, 1]
    mosaic[2::4, 1::4] = rgb_img[2::4, 1::4, 1]
    mosaic[3::4, 0::4] = rgb_img[3::4, 0::4, 1]
    mosaic[3::4, 1::4] = rgb_img[3::4, 1::4, 1]

    # Stvaranje četvrtog 2x2 bloka (Plavi)
    mosaic[2::4, 2::4] = rgb_img[2::4, 2::4, 0]
    mosaic[2::4, 3::4] = rgb_img[2::4, 3::4, 0]
    mosaic[3::4, 2::4] = rgb_img[3::4, 2::4, 0]
    mosaic[3::4, 3::4] = rgb_img[3::4, 3::4, 0]

    return mosaic.astype(rgb_img.dtype)


# Generiranje mozaika za Nonacell


def generate_nonacell_mosaic(rgb_img):
    """
    Simulacija Nonacell mozaika (3x3 blokovi iste boje). Super-piksel je 6x6.
    """
    H, W = rgb_img.shape[:2]
    mosaic = np.zeros((H, W), dtype=rgb_img.dtype)

    # Stvaranje prvog 3x3 bloka (Crveni)
    mosaic[0::6, 0::6] = rgb_img[0::6, 0::6, 2]
    mosaic[0::6, 1::6] = rgb_img[0::6, 1::6, 2]
    mosaic[0::6, 2::6] = rgb_img[0::6, 2::6, 2]
    mosaic[1::6, 0::6] = rgb_img[1::6, 0::6, 2]
    mosaic[1::6, 1::6] = rgb_img[1::6, 1::6, 2]
    mosaic[1::6, 2::6] = rgb_img[1::6, 2::6, 2]
    mosaic[2::6, 0::6] = rgb_img[2::6, 0::6, 2]
    mosaic[2::6, 1::6] = rgb_img[2::6, 1::6, 2]
    mosaic[2::6, 2::6] = rgb_img[2::6, 2::6, 2]

    # Stvaranje drugog 3x3 bloka (Zeleni)
    mosaic[0::6, 3::6] = rgb_img[0::6, 3::6, 1]
    mosaic[0::6, 4::6] = rgb_img[0::6, 4::6, 1]
    mosaic[0::6, 5::6] = rgb_img[0::6, 5::6, 1]
    mosaic[1::6, 3::6] = rgb_img[1::6, 3::6, 1]
    mosaic[1::6, 4::6] = rgb_img[1::6, 4::6, 1]
    mosaic[1::6, 5::6] = rgb_img[1::6, 5::6, 1]
    mosaic[2::6, 3::6] = rgb_img[2::6, 3::6, 1]
    mosaic[2::6, 4::6] = rgb_img[2::6, 4::6, 1]
    mosaic[2::6, 5::6] = rgb_img[2::6, 5::6, 1]

    # Stvaranje trećeg 3x3 bloka (Zeleni)
    mosaic[3::6, 0::6] = rgb_img[3::6, 0::6, 1]
    mosaic[3::6, 1::6] = rgb_img[3::6, 1::6, 1]
    mosaic[3::6, 2::6] = rgb_img[3::6, 2::6, 1]
    mosaic[4::6, 0::6] = rgb_img[4::6, 0::6, 1]
    mosaic[4::6, 1::6] = rgb_img[4::6, 1::6, 1]
    mosaic[4::6, 2::6] = rgb_img[4::6, 2::6, 1]
    mosaic[5::6, 0::6] = rgb_img[5::6, 0::6, 1]
    mosaic[5::6, 1::6] = rgb_img[5::6, 1::6, 1]
    mosaic[5::6, 2::6] = rgb_img[5::6, 2::6, 1]

    # Stvaranje četvrtog 3x3 bloka (Plavi)
    mosaic[3::6, 3::6] = rgb_img[3::6, 3::6, 0]
    mosaic[3::6, 4::6] = rgb_img[3::6, 4::6, 0]
    mosaic[3::6, 5::6] = rgb_img[3::6, 5::6, 0]
    mosaic[4::6, 3::6] = rgb_img[4::6, 3::6, 0]
    mosaic[4::6, 4::6] = rgb_img[4::6, 4::6, 0]
    mosaic[4::6, 5::6] = rgb_img[4::6, 5::6, 0]
    mosaic[5::6, 3::6] = rgb_img[5::6, 3::6, 0]
    mosaic[5::6, 4::6] = rgb_img[5::6, 4::6, 0]
    mosaic[5::6, 5::6] = rgb_img[5::6, 5::6, 0]

    return mosaic.astype(rgb_img.dtype)


# Algoritmi rekonstrukcije za Quad Bayer


# A) Izravna rekonstrukcija koja poštuje 2x2 blokove
def demosaic_direct_quad_bayer(mosaic):
    """
    Koristimo metode GaussianBlur i merge iz biblioteke cv2.
    cv2.GaussianBlur primjenjuje Gaussov filtar (zamućenje) na sliku za izglađivanje
    i smanjenje šuma.
    cv2.merge spaja tri odvojena kanala (B, G, R) u jednu višekanalnu sliku.
    """
    mosaic_float = mosaic.astype(np.float32)

    # 1. Izdvajanje kanala
    R_sparse = np.zeros_like(mosaic_float)
    G_sparse = np.zeros_like(mosaic_float)
    B_sparse = np.zeros_like(mosaic_float)

    # R Block (0,0)
    R_sparse[0::4, 0::4] = mosaic_float[0::4, 0::4]
    R_sparse[0::4, 1::4] = mosaic_float[0::4, 1::4]
    R_sparse[1::4, 0::4] = mosaic_float[1::4, 0::4]
    R_sparse[1::4, 1::4] = mosaic_float[1::4, 1::4]

    # G Block (0,2) i (2,0)
    G_sparse[0::4, 2::4] = mosaic_float[0::4, 2::4]
    G_sparse[0::4, 3::4] = mosaic_float[0::4, 3::4]
    G_sparse[1::4, 2::4] = mosaic_float[1::4, 2::4]
    G_sparse[1::4, 3::4] = mosaic_float[1::4, 3::4]

    G_sparse[2::4, 0::4] = mosaic_float[2::4, 0::4]
    G_sparse[2::4, 1::4] = mosaic_float[2::4, 1::4]
    G_sparse[3::4, 0::4] = mosaic_float[3::4, 0::4]
    G_sparse[3::4, 1::4] = mosaic_float[3::4, 1::4]

    # B Block (2,2)
    B_sparse[2::4, 2::4] = mosaic_float[2::4, 2::4]
    B_sparse[2::4, 3::4] = mosaic_float[2::4, 3::4]
    B_sparse[3::4, 2::4] = mosaic_float[3::4, 2::4]
    B_sparse[3::4, 3::4] = mosaic_float[3::4, 3::4]

    # Kreira maske prije interpolacije
    R_mask = R_sparse > 0  # True samo na R pozicijama
    G_mask = G_sparse > 0  # True samo na G pozicijama
    B_mask = B_sparse > 0  # True samo na B pozicijama

    # 2. Ovdje koristimo Gaussovo zamućenje za popunjavanje rupa
    # Veličina jezgre (kernela) je 5x5 jer neparna veličina (5) osigurava jasan centralni
    # piksel za simetrično prosječavanje, osim toga tako je jezgra dovoljno velika (4x4 + 1 piksel margine)
    # da obuhvati susjedne blokove čime se osigurava glatki prijelaz boje pri interpolaciji rupa.
    kernel_size = 5

    # Dijelimo konvoluiranu sliku s konvoluiranom maskom kako bismo dobili točan prosjek samo validnih piksela.
    R_blurred_sparse = cv2.GaussianBlur(R_sparse, (kernel_size, kernel_size), 0)
    R_blurred_mask = cv2.GaussianBlur(R_mask.astype(np.float32), (kernel_size, kernel_size), 0)
    R_blurred_mask[R_blurred_mask == 0] = 1e-6 # Izbjegavanje dijeljenja s nulom
    R_full = R_blurred_sparse / R_blurred_mask

    G_blurred_sparse = cv2.GaussianBlur(G_sparse, (kernel_size, kernel_size), 0)
    G_blurred_mask = cv2.GaussianBlur(G_mask.astype(np.float32), (kernel_size, kernel_size), 0)
    G_blurred_mask[G_blurred_mask == 0] = 1e-6
    G_full = G_blurred_sparse / G_blurred_mask

    B_blurred_sparse = cv2.GaussianBlur(B_sparse, (kernel_size, kernel_size), 0)
    B_blurred_mask = cv2.GaussianBlur(B_mask.astype(np.float32), (kernel_size, kernel_size), 0)
    B_blurred_mask[B_blurred_mask == 0] = 1e-6
    B_full = B_blurred_sparse / B_blurred_mask

    # 3. Spajanje kanala
    R_full[R_mask] = R_sparse[R_mask]
    G_full[G_mask] = G_sparse[G_mask]
    B_full[B_mask] = B_sparse[B_mask]

    # 4. Spajanje kanala u BGR format
    rgb_out = cv2.merge([B_full, G_full, R_full])

    return np.clip(rgb_out, 0, 255).astype(mosaic.dtype)


# B) Pristup pretvorbom u 'Virtualni Bayer'
def demosaic_virtual_bayer(mosaic):
    """
    Korak 1: Izvršiti 2x2 'Pixel Binning' da se dobije Bayerov mozaik niske rezolucije.
    Korak 2: Primijeniti standardni Bayerov demosaicing.
    Korak 3: Povećati rezoluciju na izvornu (ako se želi izlaz visoke rezolucije).

    Ovdje koristimo metode cv2.cvtColor i cv2.resize iz bibliotekee cv2.
    cv2.cvtColor konvertira sliku iz jednog kolornog prostora u drugi.
    cv2.resize mijenja veličinu (rezoluciju) slike.
    """
    H, W = mosaic.shape

    # 1. Pixel Binning (Spajanje 2x2 piksela iste boje)
    # Rezultirajući mozaik niske rezolucije (H/2 x W/2) je standardni Bayerov mozaik.
    # Zbog preciznosti koristimo float
    low_res_mosaic = np.zeros((H // 2, W // 2), dtype=np.float32)
    mosaic_f = mosaic.astype(np.float32)

    # U 4x4 bloku Quad Bayera, nakon binninga dobivamo 2x2 Bayerov blok (R, G; G, B)

    # R kanal (Prosjek prva 2x2 bloka)
    low_res_mosaic[0::2, 0::2] = (
        mosaic_f[0::4, 0::4]
        + mosaic_f[0::4, 1::4]
        + mosaic_f[1::4, 0::4]
        + mosaic_f[1::4, 1::4]
    ) / 4

    # G1 kanal (Prosjek drugog 2x2 bloka)
    low_res_mosaic[0::2, 1::2] = (
        mosaic_f[0::4, 2::4]
        + mosaic_f[0::4, 3::4]
        + mosaic_f[1::4, 2::4]
        + mosaic_f[1::4, 3::4]
    ) / 4

    # G2 kanal (Prosjek trećeg 2x2 bloka)
    low_res_mosaic[1::2, 0::2] = (
        mosaic_f[2::4, 0::4]
        + mosaic_f[2::4, 1::4]
        + mosaic_f[3::4, 0::4]
        + mosaic_f[3::4, 1::4]
    ) / 4

    # B kanal (Prosjek četvrtog 2x2 bloka)
    low_res_mosaic[1::2, 1::2] = (
        mosaic_f[2::4, 2::4]
        + mosaic_f[2::4, 3::4]
        + mosaic_f[3::4, 2::4]
        + mosaic_f[3::4, 3::4]
    ) / 4

    # Vraćanje tipa varijable u originalan tip
    low_res_mosaic = np.clip(low_res_mosaic, 0, 255).astype(mosaic.dtype)

    # 2. Primjena standardnog demosaicinga na Bayer niske rezolucije

    # Bayerovom uzorku, koji počinje s Plavim, Zelenim, Zelenim, Crvenim pikselom,
    # 2BGR Znači: Izlazna slika (low_res_rgb) mora biti u standardnom formatu slike s tri kanala,
    # u redoslijedu Blue, Green, Red (BGR). VNG Znači: Variable Number of Gradients. Ovo je specifičan,
    # kvalitetan algoritam za demosaicing. VNG algoritam koristi analizu lokalnih gradijenata (oštrine)
    # da bi odredio najbolji smjer interpolacije
    low_res_rgb = cv2.cvtColor(low_res_mosaic, cv2.COLOR_BayerBG2BGR_VNG)

    # 3. Povećanje rezolucije (Upscaling)
    # Povećavamo sliku na originalnu, visoku rezoluciju (2x povećanje).
    # INTER_CUBIC je parametar unutar funkcije cv2.resize()  koji definira metodu interpolacije koja će se koristiti
    # za povećanje slike (upscaling)
    recon_rgb = cv2.resize(low_res_rgb, (W, H), interpolation=cv2.INTER_CUBIC)

    return recon_rgb.astype(mosaic.dtype)


# C) Pristup Super-rezolucijom
def demosaic_super_resolution_quad_bayer(mosaic):
    """
    Hibridni pristup (Super-rezolucija): Spaja najbolje od izravne rekonstrukcije (oštrina)
    i Binning pristupa (smanjenje šuma i točnost boje).
    """

    # Metoda A je optimistična u pogledu detalja.
    direct_recon = demosaic_direct_quad_bayer(mosaic)

    # Metoda B ima bolje potiskivanje šuma.
    binned_recon = demosaic_virtual_bayer(mosaic)

    # Pretvorba u float za precizne operacije
    direct_recon_float = direct_recon.astype(np.float32)
    binned_recon_float = binned_recon.astype(np.float32)

    # 3. Izdvajamo niske frekvencije iz direktne metode
    # Gaussovo zamućenje je filtar niskih frekvencija
    blur_sigma = 2.0
    direct_low = cv2.GaussianBlur(direct_recon_float, (5, 5), blur_sigma)

    # Ovdje se izvlači komponenta oštrine i detalja iz Metoda A. Ovo su
    # informacije koje Metoda B (nakon binninga i upscalinga) gubi. Dobiveni
    # direct_high sloj služi kao "poboljšivač rezolucije".
    direct_high = direct_recon_float - direct_low

    alpha = 0.7  # Težina s kojom dodajemo visoke frekvencije (podešava oštrinu)

    # Ovdje uzimamo stabilnu, niskošumnu sliku niske rezolucije (binned_recon_float)
    # i dodajemo joj izvučene, oštre detalje iz direct_high
    hybrid = binned_recon_float + alpha * direct_high

    return np.clip(hybrid, 0, 255).astype(mosaic.dtype)


# Algoritmi rekonstrukcije za Nonacell


# A) Izravna rekonstrukcija koja poštuje 3x3 blokove
def demosaic_direct_nonacell(mosaic):
    """
    Koristimo metode GaussianBlur i merge iz biblioteke cv2.
    cv2.GaussianBlur primjenjuje Gaussov filtar (zamućenje) na sliku za izglađivanje
    i smanjenje šuma.
    cv2.merge spaja tri odvojena kanala (B, G, R) u jednu višekanalnu sliku.
    """
    mosaic_float = mosaic.astype(np.float32)

    # 1. Izdvajanje kanala
    R_sparse = np.zeros_like(mosaic_float)
    G_sparse = np.zeros_like(mosaic_float)
    B_sparse = np.zeros_like(mosaic_float)

    # R Block (0,0)
    R_sparse[0::6, 0::6] = mosaic_float[0::6, 0::6]
    R_sparse[0::6, 1::6] = mosaic_float[0::6, 1::6]
    R_sparse[0::6, 2::6] = mosaic_float[0::6, 2::6]
    R_sparse[1::6, 0::6] = mosaic_float[1::6, 0::6]
    R_sparse[1::6, 1::6] = mosaic_float[1::6, 1::6]
    R_sparse[1::6, 2::6] = mosaic_float[1::6, 2::6]
    R_sparse[2::6, 0::6] = mosaic_float[2::6, 0::6]
    R_sparse[2::6, 1::6] = mosaic_float[2::6, 1::6]
    R_sparse[2::6, 2::6] = mosaic_float[2::6, 2::6]

    # G Block (0,2) i (2,0)
    G_sparse[0::6, 3::6] = mosaic_float[0::6, 3::6]
    G_sparse[0::6, 4::6] = mosaic_float[0::6, 4::6]
    G_sparse[0::6, 5::6] = mosaic_float[0::6, 5::6]
    G_sparse[1::6, 3::6] = mosaic_float[1::6, 3::6]
    G_sparse[1::6, 4::6] = mosaic_float[1::6, 4::6]
    G_sparse[1::6, 5::6] = mosaic_float[1::6, 5::6]
    G_sparse[2::6, 3::6] = mosaic_float[2::6, 3::6]
    G_sparse[2::6, 4::6] = mosaic_float[2::6, 4::6]
    G_sparse[2::6, 5::6] = mosaic_float[2::6, 5::6]

    G_sparse[3::6, 0::6] = mosaic_float[3::6, 0::6]
    G_sparse[3::6, 1::6] = mosaic_float[3::6, 1::6]
    G_sparse[3::6, 2::6] = mosaic_float[3::6, 2::6]
    G_sparse[4::6, 0::6] = mosaic_float[4::6, 0::6]
    G_sparse[4::6, 1::6] = mosaic_float[4::6, 1::6]
    G_sparse[4::6, 2::6] = mosaic_float[4::6, 2::6]
    G_sparse[5::6, 0::6] = mosaic_float[5::6, 0::6]
    G_sparse[5::6, 1::6] = mosaic_float[5::6, 1::6]
    G_sparse[5::6, 2::6] = mosaic_float[5::6, 2::6]

    # B Block (2,2)
    B_sparse[3::6, 3::6] = mosaic_float[3::6, 3::6]
    B_sparse[3::6, 4::6] = mosaic_float[3::6, 4::6]
    B_sparse[3::6, 5::6] = mosaic_float[3::6, 5::6]
    B_sparse[4::6, 3::6] = mosaic_float[4::6, 3::6]
    B_sparse[4::6, 4::6] = mosaic_float[4::6, 4::6]
    B_sparse[4::6, 5::6] = mosaic_float[4::6, 5::6]
    B_sparse[5::6, 3::6] = mosaic_float[5::6, 3::6]
    B_sparse[5::6, 4::6] = mosaic_float[5::6, 4::6]
    B_sparse[5::6, 5::6] = mosaic_float[5::6, 5::6]

    # Kreira maske prije interpolacije
    R_mask = R_sparse > 0  # True samo na R pozicijama
    G_mask = G_sparse > 0  # True samo na G pozicijama
    B_mask = B_sparse > 0  # True samo na B pozicijama

    # 2. Ovdje koristimo Gaussovo zamućenje za popunjavanje rupa
    # Veličina jezgre (kernela) je 5x5 jer neparna veličina (5) osigurava jasan centralni
    # piksel za simetrično prosječavanje, osim toga tako je jezgra dovoljno velika (4x4 + 1 piksel margine)
    # da obuhvati susjedne blokove čime se osigurava glatki prijelaz boje pri interpolaciji rupa.
    kernel_size = 5

    # Gaussovo zamućenje širi vrijednosti iz izmjerene boje (koja nije 0) na okolne
    # nule u rijetkim kanalima (R_sparse, G_sparse, B_sparse). Kasnije se pomoću
    # maske vraćaju orginalne vrijednosti na mjesta koja nisu bila 0.

    # Dijelimo konvoluiranu sliku s konvoluiranom maskom kako bismo dobili točan prosjek samo validnih piksela.
    R_blurred_sparse = cv2.GaussianBlur(R_sparse, (kernel_size, kernel_size), 0)
    R_blurred_mask = cv2.GaussianBlur(R_mask.astype(np.float32), (kernel_size, kernel_size), 0)
    R_blurred_mask[R_blurred_mask == 0] = 1e-6
    R_full = R_blurred_sparse / R_blurred_mask

    G_blurred_sparse = cv2.GaussianBlur(G_sparse, (kernel_size, kernel_size), 0)
    G_blurred_mask = cv2.GaussianBlur(G_mask.astype(np.float32), (kernel_size, kernel_size), 0)
    G_blurred_mask[G_blurred_mask == 0] = 1e-6
    G_full = G_blurred_sparse / G_blurred_mask

    B_blurred_sparse = cv2.GaussianBlur(B_sparse, (kernel_size, kernel_size), 0)
    B_blurred_mask = cv2.GaussianBlur(B_mask.astype(np.float32), (kernel_size, kernel_size), 0)
    B_blurred_mask[B_blurred_mask == 0] = 1e-6
    B_full = B_blurred_sparse / B_blurred_mask

    # 3. Spajanje kanala
    R_full[R_mask] = R_sparse[R_mask]
    G_full[G_mask] = G_sparse[G_mask]
    B_full[B_mask] = B_sparse[B_mask]

    # 4. Spajanje kanala u BGR format
    rgb_out = cv2.merge([B_full, G_full, R_full])

    return np.clip(rgb_out, 0, 255).astype(mosaic.dtype)


# B) Pristup pretvorbom u 'Virtualni Bayer'
def demosaic_virtual_bayer_nonacell(mosaic):
    """
    Korak 1: Izvršiti 2x2 'Pixel Binning' da se dobije Bayerov mozaik niske rezolucije.
    Korak 2: Primijeniti standardni Bayerov demosaicing.
    Korak 3: Povećati rezoluciju na izvornu (ako se želi izlaz visoke rezolucije).

    Ovdje koristimo metode cv2.cvtColor i cv2.resize iz biblioteke cv2.
    cv2.cvtColor konvertira sliku iz jednog kolornog prostora u drugi.
    cv2.resize mijenja veličinu (rezoluciju) slike.
    """
    H, W = mosaic.shape

    # 1. Pixel Binning (Spajanje 2x2 piksela iste boje)
    # Rezultirajući mozaik niske rezolucije (H/2 x W/2) je standardni Bayerov mozaik.
    # Zbog preciznosti koristimo float
    low_res_mosaic = np.zeros((H // 3, W // 3), dtype=np.float32)
    mosaic_f = mosaic.astype(np.float32)

    # U 4x4 bloku Quad Bayera, nakon binninga dobivamo 2x2 Bayerov blok (R, G; G, B)

    # R kanal (Prosjek prva 2x2 bloka)
    low_res_mosaic[0::2, 0::2] = (
        mosaic_f[0::6, 0::6]
        + mosaic_f[0::6, 1::6]
        + mosaic_f[0::6, 2::6]
        + mosaic_f[1::6, 0::6]
        + mosaic_f[1::6, 1::6]
        + mosaic_f[1::6, 2::6]
        + mosaic_f[2::6, 0::6]
        + mosaic_f[2::6, 1::6]
        + mosaic_f[2::6, 2::6]
    ) / 9

    # G1 kanal (Prosjek drugog 2x2 bloka)
    low_res_mosaic[0::2, 1::2] = (
        mosaic_f[0::6, 3::6]
        + mosaic_f[0::6, 4::6]
        + mosaic_f[0::6, 5::6]
        + mosaic_f[1::6, 3::6]
        + mosaic_f[1::6, 4::6]
        + mosaic_f[1::6, 5::6]
        + mosaic_f[2::6, 3::6]
        + mosaic_f[2::6, 4::6]
        + mosaic_f[2::6, 5::6]
    ) / 9

    # G2 kanal (Prosjek trećeg 2x2 bloka)
    low_res_mosaic[1::2, 0::2] = (
        mosaic_f[3::6, 0::6]
        + mosaic_f[3::6, 1::6]
        + mosaic_f[3::6, 2::6]
        + mosaic_f[4::6, 0::6]
        + mosaic_f[4::6, 1::6]
        + mosaic_f[4::6, 2::6]
        + mosaic_f[5::6, 0::6]
        + mosaic_f[5::6, 1::6]
        + mosaic_f[5::6, 2::6]
    ) / 9

    # B kanal (Prosjek četvrtog 2x2 bloka)
    low_res_mosaic[1::2, 1::2] = (
        mosaic_f[3::6, 3::6]
        + mosaic_f[3::6, 4::6]
        + mosaic_f[3::6, 5::6]
        + mosaic_f[4::6, 3::6]
        + mosaic_f[4::6, 4::6]
        + mosaic_f[4::6, 5::6]
        + mosaic_f[5::6, 3::6]
        + mosaic_f[5::6, 4::6]
        + mosaic_f[5::6, 5::6]
    ) / 9

    # Vraćanje tipa varijable u originalan tip
    low_res_mosaic = np.clip(low_res_mosaic, 0, 255).astype(mosaic.dtype)

    # 2. Primjena standardnog demosaicinga na Bayer niske rezolucije
    # Bayerovom uzorku, koji počinje s Plavim, Zelenim, Zelenim, Crvenim pikselom,
    # 2BGR Znači: Izlazna slika (low_res_rgb) mora biti u standardnom formatu slike s tri kanala,
    # u redoslijedu Blue, Green, Red (BGR). VNG Znači: Variable Number of Gradients. Ovo je specifičan,
    # kvalitetan algoritam za demosaicing. VNG algoritam koristi analizu lokalnih gradijenata (oštrine)
    # da bi odredio najbolji smjer interpolacije
    low_res_rgb = cv2.cvtColor(low_res_mosaic, cv2.COLOR_BayerBG2BGR_VNG)

    # 3. Povećanje rezolucije (Upscaling)
    # Povećavamo sliku na originalnu, visoku rezoluciju (2x povećanje).
    # INTER_CUBIC je parametar unutar funkcije cv2.resize()  koji definira metodu interpolacije koja će se koristiti
    # za povećanje slike (upscaling)
    recon_rgb = cv2.resize(low_res_rgb, (W, H), interpolation=cv2.INTER_CUBIC)

    return recon_rgb.astype(mosaic.dtype)


# C) Pristup Super-rezolucijom
def demosaic_super_resolution_nonacell(mosaic):
    """
    Hibridni pristup (Super-rezolucija): Spaja najbolje od izravne rekonstrukcije (oštrina)
    i Binning pristupa (smanjenje šuma i točnost boje).
    """

    # Metoda A je optimistična u pogledu detalja.
    direct_recon = demosaic_direct_nonacell(mosaic)

    # Metoda B ima bolje potiskivanje šuma.
    binned_recon = demosaic_virtual_bayer_nonacell(mosaic)

    # Pretvorba u float za precizne operacije
    direct_recon_float = direct_recon.astype(np.float32)
    binned_recon_float = binned_recon.astype(np.float32)

    # 3. Izdvajamo niske frekvencije iz direktne metode
    # Gaussovo zamućenje je filtar niskih frekvencija
    blur_sigma = 2.0
    direct_low = cv2.GaussianBlur(direct_recon_float, (5, 5), blur_sigma)

    # Ovdje se izvlači komponenta oštrine i detalja iz Metoda A. Ovo su
    # informacije koje Metoda B (nakon binninga i upscalinga) gubi. Dobiveni
    # direct_high sloj služi kao "poboljšivač rezolucije".
    direct_high = direct_recon_float - direct_low

    alpha = 0.7  # Težina s kojom dodajemo visoke frekvencije (podešava oštrinu)

    # Ovdje uzimamo stabilnu, niskošumnu sliku niske rezolucije (binned_recon_float)
    # i dodajemo joj izvučene, oštre detalje iz direct_high
    hybrid = binned_recon_float + alpha * direct_high

    return np.clip(hybrid, 0, 255).astype(mosaic.dtype)




# Funkcije za prikaz slika


def show_image(img, title="Slika", cmap=None, figsize=(10, 8)):
    """Prikazuje jednu sliku."""
    plt.figure(figsize=figsize)
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_comparison(ground_truth, mosaiced, demosaiced, title="Usporedba", mosaic_type="Quad Bayer"):
    """Prikazuje ground truth, mozaiciranu i demozaiciranu sliku u jednom redu."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Ground Truth
    axes[0].imshow(cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Ground Truth\n{title}", fontsize=12)
    axes[0].axis('off')

    # Mosaiced (prikaži kao grayscale - mozaik je već 2D niz)
    axes[1].imshow(mosaiced, cmap='gray')
    axes[1].set_title(f"{mosaic_type} Mozaik\n(Vizualizacija)", fontsize=12)
    axes[1].axis('off')

    # Demosaiced
    axes[2].imshow(cv2.cvtColor(demosaiced, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Demozaicirano\n{title}", fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def load_and_prep(filename):
    """Učitava sliku i priprema je za obradu (crop na dimenzije djeljive sa 6)."""
    path = f"test_images/{filename}"
    img = cv2.imread(path)
    if img is None:
        print(f"GREŠKA: Slika {filename} nije pronađena!")
        return None
    H, W = img.shape[:2]
    return img[:H-(H%6), :W-(W%6), :]


# Funkcije za prikupljanje i vizualizaciju rezultata

def zoom_in_on_artefact(y_z, x_z, size, img, recon, title1, title2):
    h, w = img.shape[:2]
    y_z = min(y_z, h - size)
    x_z = min(x_z, w - size)

    zoom_orig = img[y_z:y_z+size, x_z:x_z+size]
    zoom_recon = recon[y_z:y_z+size, x_z:x_z+size]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(zoom_orig, cv2.COLOR_BGR2RGB))
    ax[0].set_title(title1)
    ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(zoom_recon, cv2.COLOR_BGR2RGB))
    ax[1].set_title(title2)
    ax[1].axis("off")
    plt.show()

def run_full_benchmark(image_names=None, noise_level=0.02):
    """
    Pokreće SVE metode na SVIM slikama i vraća listu rezultata.
    Ako `image_names` nije proslijeđen (None) ili je prazan, funkcija će automatski pronaći
    sve slike u direktoriju `test_images/` i proći kroz njih.
    """
    final_results = []

    # Ako nije proslijeđena lista imena, automatski pronađi slike u test_images/
    if not image_names:
        import os
        img_dir = "test_images"
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        try:
            image_names = sorted([
                f for f in os.listdir(img_dir)
                if not f.startswith('.') and os.path.splitext(f)[1].lower() in valid_exts
            ])
        except FileNotFoundError:
            print(f"GREŠKA: Direktorij {img_dir} nije pronađen.")
            return final_results

    for name in image_names:
        img = load_and_prep(name)
        if img is None: continue

        # Generiraj mozaike (jednom za svaku sliku)
        raw_qb = simulate_noise(generate_quad_bayer_mosaic(img), noise_level)
        raw_nona = simulate_noise(generate_nonacell_mosaic(img), noise_level)

        # Lista parova (Metoda, Funkcija, Mozaik)
        test_configs = [
            ("QB Direct", demosaic_direct_quad_bayer, raw_qb),
            ("QB Binning", demosaic_virtual_bayer, raw_qb),
            ("QB Super-Res", demosaic_super_resolution_quad_bayer, raw_qb),
            ("Nona Direct", demosaic_direct_nonacell, raw_nona),
            ("Nona Binning", demosaic_virtual_bayer_nonacell, raw_nona),
            ("Nona Super-Res", demosaic_super_resolution_nonacell, raw_nona)
        ]

        for label, func, raw_data in test_configs:
            recon = func(raw_data)
            res = evaluate_reconstruction(img, recon, label)
            res['Slika'] = name # Dodajemo ime slike za graf
            final_results.append(res)

    return final_results


def plot_results_comparison(noise_level=0.02):
    """
    Vizualizira usporedbu rezultata svih metoda.

    Umjesto da prima `results_list` kao argument, ova funkcija sada pokreće
    `run_full_benchmark(noise_level=...)` i koristi dobivene rezultate za crtanje.

    Args:
        noise_level: razina simuliranog šuma proslijeđena benchmarku.

    Returns:
        df, avg_df - pandas DataFrame s detaljnim i sažetim (prosječnim) metrikama.
    """
    # Pokreni benchmark interno
    results_list = run_full_benchmark(None, noise_level)

    if not results_list:
        print("Nema rezultata za prikaz. Provjerite direktorij 'test_images/' i argument noise_level.")
        return None, None

    import pandas as pd
    import seaborn as sns

    # Kreiraj DataFrame
    df = pd.DataFrame(results_list)

    # 1. Graf: PSNR Comparison
    plt.figure(figsize=(15, 6))
    sns.set_theme(style="whitegrid")

    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x="Slika", y="PSNR (Y)", hue="Metoda", palette="viridis")
    plt.title("Usporedba PSNR (Y) po slikama (Više je bolje)")
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # 2. Graf: SSIM Comparison
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x="Slika", y="SSIM (Y)", hue="Metoda", palette="magma")
    plt.title("Usporedba SSIM (Y) po slikama (Više je bolje)")
    plt.xticks(rotation=45, ha='right')
    plt.legend([],[], frameon=False)  # Hide legend on second plot

    plt.tight_layout()
    plt.show()

    # 3. Graf: Prosječne performanse
    avg_df = df.groupby("Metoda")[ ["PSNR (Y)", "SSIM (Y)", "CIEDE2000 (ΔE*)"] ].mean().reset_index()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.barplot(data=avg_df, x="Metoda", y="PSNR (Y)", hue="Metoda", palette="coolwarm", legend=False)
    plt.title("Prosječan PSNR kroz sve slike")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("PSNR (dB)")

    plt.subplot(1, 2, 2)
    sns.barplot(data=avg_df, x="Metoda", y="SSIM (Y)", hue="Metoda", palette="viridis", legend=False)
    plt.title("Prosječan SSIM kroz sve slike")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("SSIM")

    plt.tight_layout()
    plt.show()

    # 4. Tablica: Prosječne vrijednosti
    print("\n" + "="*80)
    print("PROSJEČNE VRIJEDNOSTI METRIKA")
    print("="*80)
    print(avg_df.to_string(index=False))
    print("="*80)

    return df, avg_df


def create_interactive_demo():
    """
    Kreira i prikazuje interaktivni widget za upload.
    Upravlja uploadom, obradom i vizualizacijom.
    """
    upload_btn = widgets.FileUpload(accept=".png,.jpg,.jpeg", multiple=False)
    
    method_dropdown = widgets.Dropdown(
        options=[
            ('Quad Bayer - Direct', 'qb_direct'),
            ('Quad Bayer - Binning (Virtual)', 'qb_binning'),
            ('Quad Bayer - Super Resolution', 'qb_sr'),
            ('Nonacell - Direct', 'nona_direct'),
            ('Nonacell - Binning (Virtual)', 'nona_binning'),
            ('Nonacell - Super Resolution', 'nona_sr'),
            ('Sve metode (Usporedba)', 'compare_all')
        ],
        value='qb_direct',
        description='Metoda:',
        style={'description_width': 'initial'}
    )
    
    process_btn = widgets.Button(description="Obradi", button_style='primary')
    output_view = widgets.Output()

    # Spremi uploadani sadržaj kako bi se izbjeglo ponovno učitavanje
    uploaded_content = {} 

    def on_upload_change(change):
        if not change["new"]:
            return
        
        # Očisti prethodni sadržaj
        uploaded_content.clear()
        
        # Upravljanje verzijama ipywidgets
        vals = upload_btn.value
        try:
            if isinstance(vals, dict):
                 uploaded_content['data'] = list(vals.values())[0]["content"]
            elif isinstance(vals, tuple):
                 uploaded_content['data'] = vals[0]["content"]
            else:
                 uploaded_content['data'] = vals[0]["content"]
            
            with output_view:
                clear_output()
                print("Slika učitana. Odaberite metodu i kliknite 'Obradi'.")
        except Exception as e:
            with output_view:
                print(f"Greška pri učitavanju: {e}")

    def on_process_click(b):
        if not uploaded_content:
            with output_view:
                print("Molimo prvo uploadajte sliku.")
            return

        with output_view:
            clear_output()
            print(f"Obrada slike metodom: {method_dropdown.label}...")
            try:
                content = uploaded_content['data']
                image = Image.open(io.BytesIO(content)).convert("RGB")
                img_np = np.array(image)
                # Konverzija RGB u BGR za kompatibilnost s cv2
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Provjeri dimenzije (izreži na višekratnike broja 6)
                H, W = img_cv.shape[:2]
                if H % 6 != 0 or W % 6 != 0:
                    print(f"Napomena: Slika je izrezana na dimenzije djeljive sa 6 ({H-(H%6)}x{W-(W%6)}) radi mozaiciranja.")
                    img_cv = img_cv[:H-(H%6), :W-(W%6), :]
                    img_np = img_np[:H-(H%6), :W-(W%6), :]
                
                method = method_dropdown.value

                if method == 'compare_all':
                    # Generiraj oba mozaika jednom
                    raw_qb = generate_quad_bayer_mosaic(img_cv)
                    raw_nona = generate_nonacell_mosaic(img_cv)

                    methods_config = [
                        ('Quad Bayer - Direct', 'qb_direct', raw_qb),
                        ('Quad Bayer - Binning', 'qb_binning', raw_qb),
                        ('Quad Bayer - Super Resolution', 'qb_sr', raw_qb),
                        ('Nonacell - Direct', 'nona_direct', raw_nona),
                        ('Nonacell - Binning', 'nona_binning', raw_nona),
                        ('Nonacell - Super Resolution', 'nona_sr', raw_nona)
                    ]

                    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
                    axes = axes.flatten()

                    for idx, (label, key, raw_data) in enumerate(methods_config):
                        if key == 'qb_direct': recon = demosaic_direct_quad_bayer(raw_data)
                        elif key == 'qb_binning': recon = demosaic_virtual_bayer(raw_data)
                        elif key == 'qb_sr': recon = demosaic_super_resolution_quad_bayer(raw_data)
                        elif key == 'nona_direct': recon = demosaic_direct_nonacell(raw_data)
                        elif key == 'nona_binning': recon = demosaic_virtual_bayer_nonacell(raw_data)
                        elif key == 'nona_sr': recon = demosaic_super_resolution_nonacell(raw_data)

                        res = evaluate_reconstruction(img_cv, recon, label)
                        
                        ax = axes[idx]
                        ax.imshow(cv2.cvtColor(recon, cv2.COLOR_BGR2RGB))
                        title = f"{label}\nPSNR: {res['PSNR (Y)']:.2f} dB, CIEDE2000: {res['CIEDE2000 (ΔE*)']:.2f}"
                        ax.set_title(title, fontsize=10)
                        ax.axis("off")

                    plt.tight_layout()
                    plt.show()
                    return
                
                raw = None
                recon = None
                mosaic_name = ""
                
                # Logika odabira metode
                if method.startswith('qb'):
                    raw = generate_quad_bayer_mosaic(img_cv)
                    mosaic_name = "Quad Bayer"
                    if method == 'qb_direct':
                        recon = demosaic_direct_quad_bayer(raw)
                    elif method == 'qb_binning':
                        recon = demosaic_virtual_bayer(raw)
                    elif method == 'qb_sr':
                        recon = demosaic_super_resolution_quad_bayer(raw)
                elif method.startswith('nona'):
                    raw = generate_nonacell_mosaic(img_cv)
                    mosaic_name = "Nonacell"
                    if method == 'nona_direct':
                        recon = demosaic_direct_nonacell(raw)
                    elif method == 'nona_binning':
                        recon = demosaic_virtual_bayer_nonacell(raw)
                    elif method == 'nona_sr':
                        recon = demosaic_super_resolution_nonacell(raw)
                
                if recon is None:
                    print("Greška: Metoda nije prepoznata.")
                    return

                res = evaluate_reconstruction(img_cv, recon, f"Demo {method}")
                
                # Prikaz
                # Prikazujemo sliku, i zoom-in detalja ako je slika velika
                fig, ax = plt.subplots(1, 2, figsize=(15, 8))
                ax[0].imshow(img_np)
                ax[0].set_title("Vaša Slika (Original)")
                ax[0].axis("off")
                
                ax[1].imshow(cv2.cvtColor(recon, cv2.COLOR_BGR2RGB))
                ax[1].set_title(f"{mosaic_name} - {method_dropdown.label}\nPSNR: {res['PSNR (Y)']:.2f} dB, CIEDE2000: {res['CIEDE2000 (ΔE*)']:.2f}")
                ax[1].axis("off")
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Greška pri obradi: {e}")
                import traceback
                traceback.print_exc()

    upload_btn.observe(on_upload_change, names="value")
    process_btn.on_click(on_process_click)

    # Kreiraj izgled
    box = widgets.VBox([
        widgets.HTML("<h3>Interaktivni Demo:</h3>"),
        widgets.HBox([widgets.Label("1. Odaberite sliku:"), upload_btn]),
        widgets.HBox([widgets.Label("2. Odaberite metodu:"), method_dropdown]),
        widgets.HBox([widgets.Label("3. Pokrenite:"), process_btn]),
        output_view
    ])
    display(box)
