// Ponto de entrada da página principal: navegação por abas e módulo de análise.
import { iniciarPagina } from './comum.js';
import { iniciarNavegacao, irPara } from './navegacao.js';
import { iniciarAnalise } from './analise.js';

iniciarPagina();
iniciarNavegacao();
iniciarAnalise({ irPara });
