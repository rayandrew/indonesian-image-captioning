import { future } from 'mdx-deck/themes';

const blue = '#0af';
const lightBlue = '#66fcf1';
const purple = '#0b032d';
const red = '#9a1750';
const black = '#111'
const grey = '#e3e2df';

const pandaPrismTheme /*: PrismTheme */ = {
    'plain': {
        'color': '#e6e6e6',
        'backgroundColor': '#292a2b'
    },
    'styles': [
        {
            'types': [
                'comment'
            ],
            'style': {
                'color': 'rgb(103, 107, 121)',
                'fontStyle': 'italic'
            }
        },
        {
            'types': [
                'operator',
                'variable'
            ],
            'style': {
                'color': 'rgb(230, 230, 230)'
            }
        },
        {
            'types': [
                'constant'
            ],
            'style': {
                'color': 'rgb(255, 184, 108)'
            }
        },
        {
            'types': [
                'char'
            ],
            'style': {
                'color': 'rgb(69, 169, 249)'
            }
        },
        {
            'types': [
                'string',
                'symbol',
                'inserted'
            ],
            'style': {
                'color': 'rgb(25, 249, 216)'
            }
        },
        {
            'types': [
                'punctuation',
                'tag',
                'attr-name'
            ],
            'style': {
                'color': 'rgb(255, 204, 149)'
            }
        },
        {
            'types': [
                'builtin',
                'function'
            ],
            'style': {
                'color': 'rgb(111, 193, 255)'
            }
        },
        {
            'types': [
                'keyword'
            ],
            'style': {
                'color': 'rgb(255, 154, 193)'
            }
        },
        {
            'types': [
                'changed'
            ],
            'style': {
                'color': 'rgb(255, 117, 181)'
            }
        },
        {
            'types': [
                'deleted'
            ],
            'style': {
                'color': 'rgb(255, 44, 109)'
            }
        }
    ]
};

export default {
    ...future,
    colors: {
        text: black,
        background: grey,
        blue,
        purple,
        red,
        lightBlue,
        link: red,
        pre: red,
        preBackground: '#000',
        code: red,
    },
    heading: {
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        fontWeight: 600,
        color: red
    },
    ol: {
        textAlign: 'justify',
    },
    ul: {
        textAlign: 'justify',
    },
    codeSurfer: {
        ...pandaPrismTheme,
        showNumbers: false,
    },
}