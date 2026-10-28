// Mantém este arquivo com este nome: `flutter create` gera um widget_test.dart
// de exemplo (contador) quando ele não existe, e esse exemplo não compila neste
// projeto.

import 'package:colo_ia/main.dart';
import 'package:colo_ia/servicos/gerenciador_modelo.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('abre pedindo o perfil e entra no modo paciente', (tester) async {
    await tester.pumpWidget(
      ColoIaApp(gerenciadorModelo: GerenciadorModelo.inerte()),
    );

    expect(find.text('Sou paciente'), findsOneWidget);
    expect(find.text('Sou profissional de saúde'), findsOneWidget);

    await tester.tap(find.text('Sou paciente'));
    // pump (e não pumpAndSettle): as abas têm indicadores de carregamento.
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));

    expect(find.text('Modo paciente'), findsOneWidget);
    expect(find.text('Início'), findsOneWidget);
  });

  testWidgets('o modo profissional mostra a ferramenta de análise', (tester) async {
    await tester.pumpWidget(
      ColoIaApp(gerenciadorModelo: GerenciadorModelo.inerte()),
    );

    await tester.tap(find.text('Sou profissional de saúde'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 400));

    expect(find.text('Modo profissional de saúde'), findsOneWidget);
    expect(find.text('Analisar'), findsOneWidget);
  });
}
