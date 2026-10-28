/// Conteúdo de prevenção, escrito uma vez e exibido nos dois perfis: o paciente
/// lê [Topico.texto]; o profissional lê [Topico.textoProfissional] quando ele
/// existe (e o mesmo texto do paciente quando não existe).
///
/// Os fatos de vacinação e rastreamento vêm de assets/dados/estatisticas.json;
/// mantenha os dois em sincronia ao atualizar.
library;

import 'package:flutter/material.dart';

import '../perfil.dart';

class Topico {
  const Topico({
    required this.icone,
    required this.titulo,
    required this.texto,
    this.textoProfissional,
  });

  final IconData icone;
  final String titulo;
  final String texto;
  final String? textoProfissional;

  String textoPara(Perfil perfil) =>
      perfil.ehProfissional ? (textoProfissional ?? texto) : texto;
}

class GrupoTopicos {
  const GrupoTopicos({required this.titulo, required this.topicos});

  final String titulo;
  final List<Topico> topicos;
}

const List<GrupoTopicos> prevencao = [
  GrupoTopicos(
    titulo: '1. Vacina contra o HPV',
    topicos: [
      Topico(
        icone: Icons.vaccines_outlined,
        titulo: 'A prevenção mais eficaz',
        texto: 'Protege contra os tipos de HPV que mais causam câncer. No SUS, '
            'meninas e meninos de 9 a 14 anos recebem uma única dose, antes do '
            'início da vida sexual. Quem tem até 19 anos e ainda não se '
            'vacinou pode receber conforme o estoque da região: pergunte na '
            'sua UBS.',
        textoProfissional: 'Esquema de dose única para 9 a 14 anos, meninas e '
            'meninos. Pessoas imunocomprometidas, com papilomatose '
            'respiratória recorrente ou vítimas de violência sexual seguem '
            'esquema com mais doses, quase sempre três. Não vacinados até 19 '
            'anos: conforme estoque regional.',
      ),
      Topico(
        icone: Icons.warning_amber_outlined,
        titulo: 'Vacinada? Ainda faça o exame',
        texto: 'A vacina não protege contra todos os tipos de HPV. O exame '
            'preventivo continua necessário, mesmo para quem se vacinou.',
      ),
    ],
  ),
  GrupoTopicos(
    titulo: '2. Exame preventivo',
    topicos: [
      Topico(
        icone: Icons.event_available_outlined,
        titulo: 'Papanicolau ou teste de HPV',
        texto: 'O exame é rápido e encontra alterações antes de virarem '
            'câncer. O SUS está passando do Papanicolau para o teste de '
            'DNA-HPV: nas orientações aprovadas em 2025, o rastreio é para '
            'mulheres de 25 a 49 anos, repetido a cada 5 anos quando o '
            'resultado é negativo. A troca é gradual e varia por cidade, então '
            'pergunte na UBS qual exame e qual intervalo valem para você.',
        textoProfissional: 'Orientações aprovadas pela Conitec: DNA-HPV como '
            'teste de rastreio, 25 a 49 anos, intervalo de 5 anos quando '
            'negativo; positividade para 16 ou 18 leva à colposcopia direta; '
            'autocoleta prevista para populações de difícil acesso. Na data '
            'da fonte (mar/2025) faltava a avaliação final do Ministério da '
            'Saúde: confirme o que está vigente no seu serviço.',
      ),
      Topico(
        icone: Icons.assignment_turned_in_outlined,
        titulo: 'Resultado alterado?',
        texto: 'Alteração não significa câncer. Siga a orientação da equipe '
            'de saúde e faça os exames de acompanhamento: lesões iniciais têm '
            'tratamento simples.',
      ),
    ],
  ),
  GrupoTopicos(
    titulo: '3. Hábitos e cuidados',
    topicos: [
      Topico(
        icone: Icons.shield_outlined,
        titulo: 'Preservativo',
        texto: 'Reduz o risco de infecção pelo HPV, embora não elimine '
            'totalmente, pois o vírus também pode estar na pele da região '
            'genital.',
      ),
      Topico(
        icone: Icons.smoke_free,
        titulo: 'Não fumar',
        texto: 'O tabagismo está associado a maior risco de câncer do colo do '
            'útero entre mulheres infectadas pelo HPV.',
      ),
      Topico(
        icone: Icons.medical_services_outlined,
        titulo: 'Tratar outras infecções',
        texto: 'Infecções sexualmente transmissíveis devem ser investigadas e '
            'tratadas. Peça orientação na UBS.',
      ),
    ],
  ),
  GrupoTopicos(
    titulo: 'Quando procurar atendimento',
    topicos: [
      Topico(
        icone: Icons.local_hospital_outlined,
        titulo: 'Sinais que merecem avaliação',
        texto: 'Sangramento após a relação sexual, entre as menstruações ou '
            'depois da menopausa; corrimento diferente do habitual ou com mau '
            'cheiro; dor na pelve ou durante a relação. Esses sinais têm '
            'várias causas possíveis, mas devem ser avaliados por um '
            'profissional.',
      ),
    ],
  ),
];

/// Pontos para a conversa com a paciente (aba "Orientar", só no perfil
/// profissional). São lembretes de aconselhamento, não condutas clínicas.
const List<String> pontosParaConversa = [
  'Pergunte quando foi o último exame preventivo e se o resultado foi '
      'entregue e explicado.',
  'Explique que resultado alterado não é o mesmo que câncer e que o seguimento '
      'é o que evita a progressão.',
  'Reforce que o exame vale mesmo sem sintomas e mesmo após a vacina.',
  'Lembre da vacinação de filhas e filhos de 9 a 14 anos.',
];
