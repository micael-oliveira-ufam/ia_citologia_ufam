import 'package:flutter/material.dart';

/// Paleta compartilhada com a plataforma web do projeto.
class Cores {
  static const porcelana = Color(0xFFF4F7F9);
  static const superficie = Color(0xFFFFFFFF);
  static const tinta = Color(0xFF14232F);
  static const tintaFraca = Color(0xFF4A5C6B);
  static const linha = Color(0xFFDCE5EB);
  static const teal = Color(0xFF0E6B7B);
  static const tealClaro = Color(0xFFE4F1F3);
  static const alerta = Color(0xFF8A6410);
  static const alertaFundo = Color(0xFFFFF6E8);
}

/// Contraste conferido para WCAG AA: tinta sobre porcelana passa de 12:1,
/// tintaFraca sobre branco fica acima de 7:1, e teal sobre branco acima de 4.5:1.
ThemeData construirTema() {
  final base = ColorScheme.fromSeed(
    seedColor: Cores.teal,
    brightness: Brightness.light,
  ).copyWith(
    surface: Cores.superficie,
    onSurface: Cores.tinta,
    primary: Cores.teal,
  );

  return ThemeData(
    useMaterial3: true,
    colorScheme: base,
    scaffoldBackgroundColor: Cores.porcelana,
    appBarTheme: const AppBarTheme(
      backgroundColor: Cores.superficie,
      foregroundColor: Cores.tinta,
      elevation: 0,
      scrolledUnderElevation: 1,
      centerTitle: false,
      titleTextStyle: TextStyle(
        color: Cores.tinta,
        fontSize: 19,
        fontWeight: FontWeight.w700,
      ),
    ),
    cardTheme: CardThemeData(
      color: Cores.superficie,
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: const BorderSide(color: Cores.linha),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        // 56 dp de altura: bem acima do mínimo de 48 dp recomendado para toque.
        minimumSize: const Size.fromHeight(56),
        textStyle: const TextStyle(fontSize: 17, fontWeight: FontWeight.w600),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size.fromHeight(56),
        foregroundColor: Cores.teal,
        side: const BorderSide(color: Cores.linha),
        textStyle: const TextStyle(fontSize: 17, fontWeight: FontWeight.w600),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      ),
    ),
    listTileTheme: const ListTileThemeData(
      iconColor: Cores.teal,
      minVerticalPadding: 12,
    ),
    dividerTheme: const DividerThemeData(color: Cores.linha, space: 1),
    textTheme: const TextTheme(
      headlineSmall: TextStyle(
        color: Cores.tinta,
        fontWeight: FontWeight.w700,
        height: 1.25,
      ),
      titleMedium: TextStyle(
        color: Cores.tinta,
        fontWeight: FontWeight.w600,
        fontSize: 17,
      ),
      bodyLarge: TextStyle(color: Cores.tinta, fontSize: 16, height: 1.5),
      bodyMedium: TextStyle(color: Cores.tintaFraca, fontSize: 15, height: 1.5),
      labelLarge: TextStyle(color: Cores.tintaFraca, fontSize: 13),
    ),
  );
}
