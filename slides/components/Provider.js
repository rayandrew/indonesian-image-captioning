import React from 'react';

import styled from 'styled-components';
import { updaters } from 'mdx-deck';

import { red, grey } from '../colors';

const MetaBar = styled.div([], {
    position: 'fixed',
    left: 0,
    right: 0,
    bottom: 0,
    height: '42px',
    background: grey,
    paddingLeft: '100px',
    'box-shadow': '0px -1px 5px rgba(32, 34, 39, 0.5)',
    display: 'flex',
    'align-items': 'center',
    'justify-content': 'space-between'
});

const Info = styled.div([], {
    'margin-left': '20px',
    color: red,
});

const Button = styled.div([], {
    cursor: 'pointer',
    width: '64px',
    height: '100vh',
    background: red,
});

const Previous = styled(Button)([], {
    position: 'fixed',
    top: 0,
    left: 0,
    bottom: 0,
});

const Next = styled(Button)([], {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
});

const Provider = ({ children, index, length, update }) => {
    let isPrint = false;

    const window = global;

    if (typeof window !== 'undefined' && window.navigator.userAgent.includes('Print/PDF')) {
        isPrint = true;
    }

    return (
        <React.Fragment>
            {children}
            <MetaBar>
                {!isPrint ? <Info>{index + 1} / {length}</Info> : <Info />}
            </MetaBar>
            <Previous
                role='button'
                title='Previous Slide'
                onClick={() => update(updaters.previous)}
            />
            <Next
                role='button'
                title='Next Slide'
                onClick={() => update(updaters.next)}
            />
        </React.Fragment>
    );
};

export default Provider;